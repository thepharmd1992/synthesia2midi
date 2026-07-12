from dataclasses import dataclass
from typing import Literal, Mapping


Morphology = Literal["natural", "accidental"]


@dataclass(frozen=True)
class ColorFamilyDefinition:
    number: int
    natural_slot: str
    accidental_slot: str
    midi_channel: int


COLOR_FAMILIES = (
    ColorFamilyDefinition(1, "LW", "LB", 0),
    ColorFamilyDefinition(2, "RW", "RB", 1),
    ColorFamilyDefinition(3, "COLOR_3_W", "COLOR_3_B", 2),
    ColorFamilyDefinition(4, "COLOR_4_W", "COLOR_4_B", 3),
)

SUPPORTED_EXEMPLAR_SLOTS = tuple(
    slot
    for family in COLOR_FAMILIES
    for slot in (family.natural_slot, family.accidental_slot)
)


def family_for_slot(slot: str) -> ColorFamilyDefinition | None:
    return next(
        (
            family
            for family in COLOR_FAMILIES
            if slot in (family.natural_slot, family.accidental_slot)
        ),
        None,
    )


def slots_for_family(number: int) -> tuple[str, str]:
    family = next(family for family in COLOR_FAMILIES if family.number == number)
    return family.natural_slot, family.accidental_slot


def morphology_for_slot(slot: str) -> Morphology | None:
    family = family_for_slot(slot)
    if family is None:
        return None
    return "natural" if slot == family.natural_slot else "accidental"


def exemplar_display_parts(slot: str) -> tuple[int, str]:
    family = family_for_slot(slot)
    if family is None:
        raise ValueError(f"Unsupported exemplar slot: {slot}")
    label = "Natural" if slot == family.natural_slot else "Sharp / Flat"
    return family.number, label


def active_family_numbers(
    enabled: Mapping[str, bool], colors: Mapping[str, object]
) -> tuple[int, ...]:
    active = {1}
    for family in COLOR_FAMILIES:
        slots = (family.natural_slot, family.accidental_slot)
        if any(
            enabled.get(slot, False) or colors.get(slot) is not None
            for slot in slots
        ):
            active.add(family.number)
    return tuple(sorted(active))
