"""Tier enum — mirrors autogen.net Tier enum (Free / Testing / Regular / Premium)."""

from __future__ import annotations

from enum import StrEnum


class Tier(StrEnum):
    """Usage tier — determines rate limits, feature access, and storage quotas.

    Mirrors autogen.net's Tier enum (Free / Testing / Regular / Premium).
    """

    FREE = "Free"
    TESTING = "Testing"
    REGULAR = "Regular"
    PREMIUM = "Premium"
