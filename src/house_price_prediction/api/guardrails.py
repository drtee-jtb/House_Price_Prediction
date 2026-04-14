from __future__ import annotations

import re

from house_price_prediction.config import Settings
from house_price_prediction.domain.contracts.prediction_contracts import AddressPayload

_PO_BOX_PATTERN = re.compile(r"\bP\s*\.?\s*O\s*\.?\s*BOX\b", re.IGNORECASE)


class RequestGuardrailError(ValueError):
    pass


def validate_address_payload(payload: AddressPayload, settings: Settings) -> None:
    if _PO_BOX_PATTERN.search(payload.address_line_1):
        raise RequestGuardrailError("Street address is required; PO boxes are not supported.")

    if _uses_us_only_providers(settings) and payload.country.strip().upper() != "US":
        raise RequestGuardrailError(
            "Configured public providers currently support only US addresses."
        )


def _uses_us_only_providers(settings: Settings) -> bool:
    return settings.geocoding_provider.strip().lower().startswith("free") or settings.property_data_provider.strip().lower().startswith("free")