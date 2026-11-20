"""DynamoDB state for production. Two tables:

* state (pk, sk): team policies, client-to-team mapping, rate-limit windows, monthly spend, response cache.
  Items that expire carry a ``ttl`` attribute (DynamoDB TTL enabled on it).
* audit (pk=team_id, sk="<iso time>#<request id>"): one item per request, kept ~13 months.

Limits use a single conditional ``UpdateItem`` (ADD with a condition on the current value), so the
check and the increment are one atomic operation across any number of Lambda containers.
"""
from __future__ import annotations

import json
import time

from ..teams import TeamPolicy

AUDIT_RETENTION_S = 400 * 24 * 3600


def _s(v: str) -> dict:
    return {"S": v}


def _n(v: int | float) -> dict:
    return {"N": str(v)}


class DynamoStore:
    def __init__(self, state_table: str, audit_table: str, client=None, region: str | None = None):
        import boto3

        self.state = state_table
        self.audit = audit_table
        self.ddb = client or boto3.client("dynamodb", region_name=region)

    # --- teams -------------------------------------------------------------------------
    def put_team(self, policy: TeamPolicy) -> None:
        old = self.get_team(policy.team_id)
        self.ddb.put_item(TableName=self.state, Item={"pk": _s(f"TEAM#{policy.team_id}"), "sk": _s("POLICY"),
                                                     "body": _s(json.dumps(policy.to_dict()))})
        for cid in policy.client_ids:
            self.ddb.put_item(TableName=self.state, Item={"pk": _s(f"CLIENT#{cid}"), "sk": _s("TEAM"), "team_id": _s(policy.team_id)})
        for cid in set(old.client_ids if old else []) - set(policy.client_ids):
            self.ddb.delete_item(TableName=self.state, Key={"pk": _s(f"CLIENT#{cid}"), "sk": _s("TEAM")})

    def get_team(self, team_id: str) -> TeamPolicy | None:
        item = self.ddb.get_item(TableName=self.state, Key={"pk": _s(f"TEAM#{team_id}"), "sk": _s("POLICY")},
                                 ConsistentRead=True).get("Item")
        return TeamPolicy.from_dict(json.loads(item["body"]["S"])) if item else None

    def team_for_client(self, client_id: str) -> TeamPolicy | None:
        item = self.ddb.get_item(TableName=self.state, Key={"pk": _s(f"CLIENT#{client_id}"), "sk": _s("TEAM")}).get("Item")
        return self.get_team(item["team_id"]["S"]) if item else None

    def list_teams(self) -> list[TeamPolicy]:
        teams = []
        for page in self.ddb.get_paginator("scan").paginate(
                TableName=self.state, FilterExpression="sk = :p", ExpressionAttributeValues={":p": _s("POLICY")}):
            teams += [TeamPolicy.from_dict(json.loads(i["body"]["S"])) for i in page["Items"]]
        return sorted(teams, key=lambda t: t.team_id)

    # --- limits ------------------------------------------------------------------------
    def _conditional_add(self, key: dict, amount: int, ceiling: int, ttl: int | None) -> tuple[bool, int]:
        if amount > ceiling:
            return False, self._value(key)
        update = "ADD #v :a" + (" SET #t = if_not_exists(#t, :ttl)" if ttl else "")
        values = {":a": _n(amount), ":max": _n(ceiling - amount)}
        names = {"#v": "value"}
        if ttl:
            values[":ttl"] = _n(ttl)
            names["#t"] = "ttl"
        try:
            out = self.ddb.update_item(TableName=self.state, Key=key, UpdateExpression=update,
                                       ConditionExpression="attribute_not_exists(#v) OR #v <= :max",
                                       ExpressionAttributeNames=names, ExpressionAttributeValues=values,
                                       ReturnValues="UPDATED_NEW", ReturnValuesOnConditionCheckFailure="ALL_OLD")
            return True, int(out["Attributes"]["value"]["N"])
        except self.ddb.exceptions.ConditionalCheckFailedException as e:
            old = e.response.get("Item", {})
            return False, int(old["value"]["N"]) if "value" in old else 0

    def _value(self, key: dict) -> int:
        item = self.ddb.get_item(TableName=self.state, Key=key, ConsistentRead=True).get("Item")
        return int(item["value"]["N"]) if item else 0

    def hit_window(self, key: str, amount: int, limit: int, ttl_s: int) -> tuple[bool, int]:
        return self._conditional_add({"pk": _s(f"LIMIT#{key}"), "sk": _s("W")}, amount, limit, int(time.time()) + ttl_s)

    def reserve_spend(self, team_id: str, month: str, amount: int, budget: int) -> tuple[bool, int]:
        return self._conditional_add({"pk": _s(f"SPEND#{team_id}"), "sk": _s(month)}, amount, budget, None)

    def adjust_spend(self, team_id: str, month: str, delta: int) -> int:
        out = self.ddb.update_item(TableName=self.state, Key={"pk": _s(f"SPEND#{team_id}"), "sk": _s(month)},
                                   UpdateExpression="ADD #v :d", ExpressionAttributeNames={"#v": "value"},
                                   ExpressionAttributeValues={":d": _n(delta)}, ReturnValues="UPDATED_NEW")
        return int(out["Attributes"]["value"]["N"])

    def get_spend(self, team_id: str, month: str) -> int:
        return self._value({"pk": _s(f"SPEND#{team_id}"), "sk": _s(month)})

    # --- cache -------------------------------------------------------------------------
    def cache_get(self, key: str) -> dict | None:
        item = self.ddb.get_item(TableName=self.state, Key={"pk": _s(f"CACHE#{key}"), "sk": _s("R")}).get("Item")
        if not item or int(item["ttl"]["N"]) <= time.time():   # TTL deletion is lazy, so check expiry here
            return None
        return json.loads(item["body"]["S"])

    def cache_put(self, key: str, value: dict, ttl_s: int) -> None:
        self.ddb.put_item(TableName=self.state, Item={"pk": _s(f"CACHE#{key}"), "sk": _s("R"), "body": _s(json.dumps(value)),
                                                     "ttl": _n(int(time.time()) + ttl_s)})

    # --- audit -------------------------------------------------------------------------
    def write_audit(self, record: dict) -> None:
        self.ddb.put_item(TableName=self.audit, Item={
            "pk": _s(record.get("team_id") or "_unauthenticated"), "sk": _s(f"{record['ts']}#{record['request_id']}"),
            "body": _s(json.dumps(record)), "ttl": _n(int(time.time()) + AUDIT_RETENTION_S)})

    def audit_records(self, team_id: str | None = None, since: str | None = None) -> list[dict]:
        if team_id:
            kwargs = {"TableName": self.audit, "KeyConditionExpression": "pk = :t" + (" AND sk >= :s" if since else ""),
                      "ExpressionAttributeValues": {":t": _s(team_id), **({":s": _s(since)} if since else {})}}
            pages = self.ddb.get_paginator("query").paginate(**kwargs)
        else:
            kwargs = {"TableName": self.audit}
            if since:
                kwargs.update(FilterExpression="sk >= :s", ExpressionAttributeValues={":s": _s(since)})
            pages = self.ddb.get_paginator("scan").paginate(**kwargs)
        out = [json.loads(i["body"]["S"]) for p in pages for i in p["Items"]]
        return sorted(out, key=lambda r: r["ts"])
