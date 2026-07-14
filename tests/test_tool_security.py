"""Security tests for the invoice tool layer.

Verifies the two hard guarantees added after security review:
1. Identity binding — customer_id is injected from graph state, hidden
   from the LLM-facing schema, and tools refuse without it.
2. Ownership filtering — invoice-scoped lookups never return another
   customer's data (no IDOR).

No API key required.
"""

import json

import pytest

from src.tools import (
    INVOICE_TOOLS,
    get_invoice_line_items,
    get_invoices_by_customer_sorted_by_date,
    get_purchased_tracks_by_customer,
    get_support_rep_by_invoice,
)

# Invoice 1 belongs to customer 2 in Chinook.
FOREIGN_CUSTOMER = "5"
OWNER_CUSTOMER = "2"
INVOICE_OF_CUSTOMER_2 = "1"


class TestLLMSchemaHidesCustomerId:
    @pytest.mark.parametrize("tool_fn", INVOICE_TOOLS, ids=lambda t: t.name)
    def test_customer_id_not_in_llm_schema(self, tool_fn):
        """The model must not be able to pass or alter customer_id."""
        properties = tool_fn.tool_call_schema.model_json_schema().get("properties", {})
        assert "customer_id" not in properties


class TestUnverifiedAccessRefused:
    @pytest.mark.parametrize(
        "tool_fn,args",
        [
            (get_invoices_by_customer_sorted_by_date, {}),
            (get_purchased_tracks_by_customer, {}),
            (get_support_rep_by_invoice, {"invoice_id": "1"}),
            (get_invoice_line_items, {"invoice_id": "1"}),
        ],
        ids=lambda x: x.name if hasattr(x, "name") else "",
    )
    def test_none_customer_id_refused(self, tool_fn, args):
        """With no verified identity in state, every tool must refuse."""
        result = tool_fn.invoke({**args, "customer_id": None})
        payload = json.loads(result)
        assert "error" in payload
        assert "not verified" in payload["error"].lower()


class TestOwnershipFiltering:
    def test_support_rep_denied_for_foreign_invoice(self):
        """Customer 5 must not read invoice 1 (owned by customer 2)."""
        result = get_support_rep_by_invoice.invoke(
            {"invoice_id": INVOICE_OF_CUSTOMER_2, "customer_id": FOREIGN_CUSTOMER}
        )
        payload = json.loads(result)
        assert "message" in payload  # not-found, no data leaked
        assert "SupportRepName" not in json.dumps(payload)

    def test_line_items_denied_for_foreign_invoice(self):
        result = get_invoice_line_items.invoke(
            {"invoice_id": INVOICE_OF_CUSTOMER_2, "customer_id": FOREIGN_CUSTOMER}
        )
        payload = json.loads(result)
        assert "message" in payload
        assert "TrackName" not in json.dumps(payload)

    def test_owner_still_allowed(self):
        """The ownership filter must not block the legitimate owner."""
        rep = json.loads(
            get_support_rep_by_invoice.invoke(
                {"invoice_id": INVOICE_OF_CUSTOMER_2, "customer_id": OWNER_CUSTOMER}
            )
        )
        items = json.loads(
            get_invoice_line_items.invoke(
                {"invoice_id": INVOICE_OF_CUSTOMER_2, "customer_id": OWNER_CUSTOMER}
            )
        )
        assert rep[0]["CustomerId"] == 2
        assert len(items) == 2

    def test_no_employee_phone_in_output(self):
        """Employee PII is minimized: no phone number in rep output."""
        result = get_support_rep_by_invoice.invoke(
            {"invoice_id": INVOICE_OF_CUSTOMER_2, "customer_id": OWNER_CUSTOMER}
        )
        assert "SupportRepPhone" not in result

    def test_invoice_list_omits_full_billing_address(self):
        """Invoice summaries should not expose unnecessary street addresses."""
        result = get_invoices_by_customer_sorted_by_date.invoke(
            {"customer_id": OWNER_CUSTOMER}
        )
        rows = json.loads(result)
        assert rows
        assert "BillingAddress" not in rows[0]
