"""Quirk for ZLinky_TIC."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
import re
import time
from typing import TYPE_CHECKING, Any, Final, NamedTuple

from zha.application import Platform
from zha.zigbee.device import DeviceEntityRemovedEvent
from zigpy.exceptions import ZigbeeException
import zigpy.types as t
from zigpy.zcl import ClusterType
from zigpy.zcl.clusters.general import PowerConfiguration, Time
from zigpy.zcl.clusters.homeautomation import ElectricalMeasurement
from zigpy.zcl.clusters.smartenergy import Metering
from zigpy.zcl.foundation import BaseAttributeDefs, ZCLAttributeDef

from zhaquirks.builder import (
    BinarySensorDeviceClass,
    EntityPlatform,
    EntityType,
    QuirkBuilder,
    ReportingConfig,
    SensorDeviceClass,
    SensorStateClass,
    UnitOfApparentPower,
    UnitOfElectricCurrent,
    UnitOfEnergy,
    UnitOfPower,
    UnitOfTime,
)
from zhaquirks.builder.device import QuirkV2Device
from zhaquirks.builder.metadata import DeviceAlertLevel, DeviceAlertMetadata
from zhaquirks.clusters import CustomCluster
from zhaquirks.lixee import LIXEE, ZLINKY_MANUFACTURER_CLUSTER_ID
from zhaquirks.tuya import TuyaManufCluster

if TYPE_CHECKING:
    from zha.application.platforms import BaseEntity

# Firmware versions (OTA `current_file_version`) from which the device exposes
# PowerConfiguration (v12), then Time and Tuya (v14). The legacy v1 quirk told
# them apart by cluster signature instead.
_FW_V12 = 0x0000000C
_FW_V14 = 0x0000000E
_FW_V15 = 0x0000000F


class LinkyMode(t.enum8):
    """TIC mode reported by the ZLinky.

    Bit 0: standard (1) / historique (0), bit 1: three-phase, bit 2: producer.
    """

    historique_single_phase = 0x00
    standard_single_phase = 0x01
    historique_three_phase = 0x02
    standard_three_phase = 0x03
    standard_single_phase_producer = 0x05
    standard_three_phase_producer = 0x07


LINKY_MODE_STANDARD = 0x01
LINKY_MODE_THREE_PHASE = 0x02
LINKY_MODE_PRODUCER = 0x04

# Interval (seconds) to check that the entities still match the meter. Only
# compares cached values, so it is cheap while nothing changed.
_CONFIG_CHECK_INTERVAL = 60 * 60
# Emitted by the manufacturer cluster for `ZLinkyTICDevice` to recompute the entities
ENTITIES_CHANGED_EVENT: Final = "zlinky_entities_changed"
# Not an attribute id: marks single-index contracts among the irrelevant ids, so
# that a contract change is detected like the other configuration changes
SINGLE_INDEX_KEY: Final = -1
# Same for the supplier indexes: `tier_key(tier)` is irrelevant while the meter
# reports no energy on that index, so only the indexes the contract uses get
# an entity. The firmware reports 0 for the indexes the meter does not send, so
# 0 cannot be told apart from an index not used yet: its entity is created once
# it counts energy (at the next configuration check).
_TIERS: Final = range(1, 11)
# Delay (seconds) before retrying a failed refresh or entity recomputation
_RETRY_DELAY = 10 * 60
# Maximum interval (seconds) between energy counter reports
_ENERGY_REPORTING_MAX_INTERVAL = 15 * 60


def tier_key(tier: int) -> int:
    """Return the pseudo attribute id marking supplier index `tier` as unused."""
    return -100 - tier


# Same for the slots of the day profiles: `profile_slot_key(attr_id, slot)` is
# irrelevant while the profile has fewer slots, see `_day_profile_slots`. The
# firmware only sends the first 8 blocks, so there are at most 8 slots.
_PROFILE_SLOTS: Final = range(1, 9)


def profile_slot_key(attr_id: int, slot: int) -> int:
    """Return the pseudo attribute id marking slot `slot` of a day profile as absent."""
    return -1000 - attr_id * 16 - slot


class TempoColor(t.enum8):
    """Tempo day color, as announced in the status register."""

    none = 0x00
    blue = 0x01
    white = 0x02
    red = 0x03


class ZLinkyTICManufacturerCluster(CustomCluster):
    """ZLinky_TIC manufacturer-specific cluster (0xFF66).

    Exposes the LiXee-specific TIC (Télé-Information Client) frames. The
    attribute comments below are in French to match the reference
    documentation, see
    https://github.com/fairecasoimeme/Zlinky_TIC/tree/v9.0#synth%C3%A8se-d%C3%A9veloppeur
    and https://github.com/fairecasoimeme/Zlinky_TIC/blob/v9.0/ZLinky/Source/LixeeCluster.h

    Most attributes of this cluster are read-only and not reportable (e.g.
    DATE, STGE, RELAIS, LTARF), so they are periodically re-read whenever the
    device sends any report, see `refresh_non_reportable`.

    The firmware answers for every attribute whatever the meter configuration,
    so attributes irrelevant to the TIC mode, phases, production or contract
    are reported as unsupported to keep ZHA from creating their entities, see
    `is_attribute_unsupported`. As the configuration can change (e.g. when the
    TIC is switched to standard mode), it is checked hourly and when the TIC mode
    changes, and ZHA is asked to recompute the entities on change, see
    `ZLinkyTICDevice`.
    """

    cluster_id = ZLINKY_MANUFACTURER_CLUSTER_ID
    name = "ZLinky_TIC Manufacturer specific"
    ep_attribute = "zlinky_manufacturer_specific"

    class AttributeDefs(BaseAttributeDefs):
        """Manufacturer-specific attribute definitions."""

        # Historical mode: OPTARIF "Option tarifaire" / String 4 car
        # Standard mode: NGTF "Nom du calendrier tarifaire fournisseur" / String 16 car
        hist_tariff_option_or_std_supplier_price_schedule_name: Final = ZCLAttributeDef(
            id=0x0000, type=t.LimitedCharString(16), is_manufacturer_specific=True
        )
        # Historical mode: DEMAIN "Couleur du lendemain" / String 4 car
        hist_tomorrow_color: Final = ZCLAttributeDef(
            id=0x0001, type=t.LimitedCharString(4), is_manufacturer_specific=True
        )
        # Historical mode: HHPHC "Horaire Heure Pleines Heures Creuses" / Uint8 1 car
        hist_schedule_peak_hours_off_peak_hours: Final = ZCLAttributeDef(
            id=0x0002, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Historical mode: PPOT "Présence des potentiels" (Triphasé) / Uint8 2 car
        hist_potentials_presence: Final = ZCLAttributeDef(
            id=0x0003, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Historical mode: PEJP "Préavis début EJP(30min)" / Uint8 2 car
        hist_ejp_start_notice: Final = ZCLAttributeDef(
            id=0x0004, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Historical mode: ADPS "Avertissement de Dépassement De Puissance Souscrite" / Uint16 3 car
        hist_subscribed_power_exceeding_warning: Final = ZCLAttributeDef(
            id=0x0005, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Historical mode: ADIR1 "Avertissement de Dépassement D'intensité phase 1" / Uint16 3 car
        hist_current_exceeding_warning_phase_1: Final = ZCLAttributeDef(
            id=0x0006, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Historical mode: ADIR2 "Avertissement de Dépassement D'intensité phase 2" / Uint16 3 car
        hist_current_exceeding_warning_phase_2: Final = ZCLAttributeDef(
            id=0x0007, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Historical mode: ADIR3 "Avertissement de Dépassement D'intensité phase 3" / Uint16 3 car
        hist_current_exceeding_warning_phase_3: Final = ZCLAttributeDef(
            id=0x0008, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Historical mode: MOTDETAT "Etat du Linky (From V13)" / String 6 car
        linky_status: Final = ZCLAttributeDef(
            id=0x0009, type=t.LimitedCharString(6), is_manufacturer_specific=True
        )
        # Historical and standard mode: "Tariff Period (From V15)" / String 16 car
        linky_tariff_period: Final = ZCLAttributeDef(
            id=0x0010, type=t.LimitedCharString(16), is_manufacturer_specific=True
        )
        # Historical and Standard mode: "Linky acquisition time (From V7)" / Uint8 1 car
        linky_acquisition_time: Final = ZCLAttributeDef(
            id=0x0100, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Standard mode: LTARF "Libellé tarif fournisseur en cours" / String 16 car
        std_current_supplier_price_description: Final = ZCLAttributeDef(
            id=0x0200, type=t.LimitedCharString(16), is_manufacturer_specific=True
        )
        # Standard mode: NTARF "Numéro de l'index tarifaire en cours" / Uint8 2 car
        std_current_tariff_index_number: Final = ZCLAttributeDef(
            id=0x0201, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Standard mode: DATE "Date et heure courant" / String 13 car (SAAMMJJhhmmss)
        std_current_date_and_time: Final = ZCLAttributeDef(
            id=0x0202, type=t.LimitedCharString(13), is_manufacturer_specific=True
        )
        # Standard mode: EASD01 "Energie active soutirée Distributeur, index 01" / Uint32 9 car
        std_active_energy_withdrawn_distributor_index_01: Final = ZCLAttributeDef(
            id=0x0203, type=t.uint32_t, is_manufacturer_specific=True
        )
        # Standard mode: EASD02 "Energie active soutirée Distributeur, index 02" / Uint32 9 car
        std_active_energy_withdrawn_distributor_index_02: Final = ZCLAttributeDef(
            id=0x0204, type=t.uint32_t, is_manufacturer_specific=True
        )
        # Standard mode: EASD03 "Energie active soutirée Distributeur, index 03" / Uint32 9 car
        std_active_energy_withdrawn_distributor_index_03: Final = ZCLAttributeDef(
            id=0x0205, type=t.uint32_t, is_manufacturer_specific=True
        )
        # Standard mode: EASD04 "Energie active soutirée Distributeur, index 04" / Uint32 9 car
        std_active_energy_withdrawn_distributor_index_04: Final = ZCLAttributeDef(
            id=0x0206, type=t.uint32_t, is_manufacturer_specific=True
        )
        # Standard mode: SINSTI "Puissance app. Instantanée injectée" (Production) / Uint16 5 car
        std_apparent_power_injected_instantaneous: Final = ZCLAttributeDef(
            id=0x0207, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Standard mode: SMAXIN "Puissance app max. injectée n" (Production) / Uint16 5 car
        std_apparent_power_injected_max: Final = ZCLAttributeDef(
            id=0x0208, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Standard mode: SMAXIN-1 "Puissance app max. injectée n-1" (Production) / Uint16 5 car
        std_apparent_power_injected_max_1: Final = ZCLAttributeDef(
            id=0x0209, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Standard mode: CCAIN "Point n de la courbe de charge active injectée" (Production) / Uint16 5 car
        std_injected_active_load_curve_point_n: Final = ZCLAttributeDef(
            id=0x0210, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Standard mode: CCAIN-1 "Point n-1 de la courbe de charge active injectée" (Production) / Uint16 5 car
        std_injected_active_load_curve_point_n_1: Final = ZCLAttributeDef(
            id=0x0211, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Standard mode: SMAXSN-1 "Puissance app. max. soutirée n-1" (Monophasé) / Uint16 5 car
        # Standard mode: SMAXSN1-1 "Puissance app. max. soutirée n-1 ph.1" (Triphasé) / Uint16 5 car
        std_apparent_power_withdrawn_max_phase_1_n_1: Final = ZCLAttributeDef(
            id=0x0212, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Standard mode: SMAXSN2-1 "Puissance app. max. soutirée n-1 ph. 2" (Triphasé) / Uint16 5 car
        std_apparent_power_withdrawn_max_phase_2_n_1: Final = ZCLAttributeDef(
            id=0x0213, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Standard mode: SMAXSN3-1 "Puissance app. max. soutirée n-1 ph. 3" (Triphasé) / Uint16 5 car
        std_apparent_power_withdrawn_max_phase_3_n_1: Final = ZCLAttributeDef(
            id=0x0214, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Standard mode: MSG1 "Message court" / String 32 car
        std_message_short: Final = ZCLAttributeDef(
            id=0x0215, type=t.LimitedCharString(32), is_manufacturer_specific=True
        )
        # Standard mode: MSG2 "Message ultra court" / String 16 car
        std_message_ultra_short: Final = ZCLAttributeDef(
            id=0x0216, type=t.LimitedCharString(16), is_manufacturer_specific=True
        )
        # Standard mode: STGE "Registre de Statuts" / Octet string 8 car /* codespell:ignore */
        # Sent as an octet string (0x41): 8 hex characters, most significant byte first.
        std_status_register: Final = ZCLAttributeDef(
            id=0x0217, type=t.LVBytes, is_manufacturer_specific=True
        )
        # Standard mode: DPM1 "Début Pointe Mobile 1" / Uint8 2 car
        std_mobile_peak_start_1: Final = ZCLAttributeDef(
            id=0x0218, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Standard mode: FPM1 "Fin Pointe Mobile 1" / Uint8 2 car
        std_mobile_peak_end_1: Final = ZCLAttributeDef(
            id=0x0219, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Standard mode: DPM2 "Début Pointe Mobile 2" / Uint8 2 car
        std_mobile_peak_start_2: Final = ZCLAttributeDef(
            id=0x0220, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Standard mode: FPM2 "Fin Pointe Mobile 2" / Uint8 2 car
        std_mobile_peak_end_2: Final = ZCLAttributeDef(
            id=0x0221, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Standard mode: DPM3 "Début Pointe Mobile 3" / Uint8 2 car
        std_mobile_peak_start_3: Final = ZCLAttributeDef(
            id=0x0222, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Standard mode: FPM3 "Fin Pointe Mobile 3" / Uint8 2 car
        std_mobile_peak_end_3: Final = ZCLAttributeDef(
            id=0x0223, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Standard mode: RELAIS "RELAIS" / Uint16 3 car
        std_relay: Final = ZCLAttributeDef(
            id=0x0224, type=t.uint16_t, is_manufacturer_specific=True
        )
        # Standard mode: NJOURF "Numéro du jour en cours calendrier fournisseur" / Uint8 2 car
        std_supplier_calendar_current_day_number: Final = ZCLAttributeDef(
            id=0x0225, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Standard mode: NJOURF+1 "Numéro du prochain jour calendrier fournisseur" / Uint8 2 car
        std_supplier_calendar_next_day_number: Final = ZCLAttributeDef(
            id=0x0226, type=t.uint8_t, is_manufacturer_specific=True
        )
        # Standard mode: PJOURF+1 "Profil du prochain jour calendrier fournisseur" / String 98 car
        std_supplier_calendar_next_day_profile: Final = ZCLAttributeDef(
            id=0x0227, type=t.LimitedCharString(98), is_manufacturer_specific=True
        )
        # Standard mode: PPOINTE1 "Profil du prochain jour de pointe" / String 98 car
        std_next_peak_day_profile: Final = ZCLAttributeDef(
            id=0x0228, type=t.LimitedCharString(98), is_manufacturer_specific=True
        )
        # Historical and Standard mode: - "Linky Mode (From V4)" / Uint8 1 car
        linky_mode: Final = ZCLAttributeDef(
            id=0x0300, type=LinkyMode, is_manufacturer_specific=True
        )
        # Virtual attributes decoded from STGE, never read from the device
        tempo_color_today: Final = ZCLAttributeDef(
            id=0xF000, type=TempoColor, access="rp", is_manufacturer_specific=True
        )
        tempo_color_tomorrow: Final = ZCLAttributeDef(
            id=0xF001, type=TempoColor, access="rp", is_manufacturer_specific=True
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init."""
        # Must exist before the parent init, which may populate the cache
        self._last_refresh: dict[int, float] = {}
        self._last_config_check: float | None = None
        # Set when the TIC mode changed, to check the configuration once the
        # attributes of the new mode are read
        self._mode_changed = False
        # Attributes hidden when ZHA created the entities
        self._entities_irrelevant_ids: set[int] | None = None
        # Set when updating the entities failed, to update them again
        self._entities_outdated = False
        super().__init__(*args, **kwargs)

    def _update_attribute(self, attrid: Any, value: Any) -> None:
        """Update attribute, then use the update as a cue to refresh other ones."""
        attr_id = _attribute_id(self, attrid)
        previous_mode = self.get(self.AttributeDefs.linky_mode.id)
        previous_profile = (
            self.get(attr_id) if attr_id in _DAY_PROFILE_PREFIXES else None
        )
        super()._update_attribute(attrid, value)

        if (
            attr_id == self.AttributeDefs.linky_mode.id
            and previous_mode is not None
            and value != previous_mode
        ):
            # The attributes of the other mode, e.g. OPTARIF and NGTF sharing
            # 0x0000, are stale: read them all again before checking
            self.info("TIC mode changed from %s to %s", previous_mode, value)
            self._last_refresh.clear()
            self._mode_changed = True

        if attr_id == self.AttributeDefs.std_status_register.id:
            register = _parse_status_register(value)
            if register is not None:
                super()._update_attribute(
                    self.AttributeDefs.tempo_color_today.id,
                    TempoColor((register >> 24) & 0x03),
                )
                super()._update_attribute(
                    self.AttributeDefs.tempo_color_tomorrow.id,
                    TempoColor((register >> 26) & 0x03),
                )

        # The slots of the next day can change every day: create their
        # entities without waiting for the next configuration check
        if attr_id in _DAY_PROFILE_PREFIXES and value != previous_profile:
            self._check_meter_configuration()

        self.refresh_non_reportable()

    async def read_attributes(
        self, attributes: list[Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Read attributes, serving the virtual ones from the cache."""
        virtual = [
            attr
            for attr in attributes
            if _attribute_id(self, attr) in _VIRTUAL_ATTRIBUTE_IDS
        ]
        if not virtual:
            return await super().read_attributes(attributes, *args, **kwargs)

        remote = [attr for attr in attributes if attr not in virtual]
        success, failure = (
            await super().read_attributes(remote, *args, **kwargs)
            if remote
            else ({}, {})
        )
        for attr in virtual:
            if (value := self.get(attr)) is not None:
                success[attr] = value
        return success, failure

    def is_attribute_unsupported(self, attr: Any) -> bool:
        """Report attributes irrelevant to this meter as unsupported.

        The first call comes from ZHA creating the entities (the refresh uses
        `_is_hidden`), so it records the configuration they were created for.
        """
        irrelevant = self._irrelevant_attribute_ids()
        if self._entities_irrelevant_ids is None:
            self._entities_irrelevant_ids = irrelevant
        return self._is_hidden(attr, irrelevant)

    def _is_hidden(self, attr: Any, irrelevant: set[int]) -> bool:
        """Return whether an attribute is irrelevant or unsupported by the device."""
        if _attribute_id(self, attr) in irrelevant:
            return True
        return super().is_attribute_unsupported(attr)

    def meter_mode(self) -> int | None:
        """Return the TIC mode, None while unknown.

        The firmware reports 0 (historique single-phase) until it decoded a
        first frame, so 0 is only trusted once OPTARIF, which every historique
        frame carries, has been received.
        """
        mode = self.get(self.AttributeDefs.linky_mode.id)
        if mode is None or (not mode and self._tariff_option() is None):
            return None
        return mode

    def _check_meter_configuration(self) -> None:
        """Make ZHA recompute the entities if they do not match the meter anymore."""
        if (
            self._entities_irrelevant_ids is None
            or self.meter_mode() is None
            or (
                not self._entities_outdated
                and self._irrelevant_attribute_ids() == self._entities_irrelevant_ids
            )
        ):
            return
        self.info("Meter configuration changed, updating entities")
        # Recorded again when ZHA re-evaluates the entities
        self._entities_irrelevant_ids = None
        self._entities_outdated = False
        self.emit(ENTITIES_CHANGED_EVENT)

    def retry_entities_update(self) -> None:
        """Update the entities again soon, after a failed update."""
        self._entities_outdated = True
        # Not recorded if the update failed before ZHA evaluated the entities
        if self._entities_irrelevant_ids is None:
            self._entities_irrelevant_ids = set()
        self._last_config_check = (
            time.monotonic() - _CONFIG_CHECK_INTERVAL + _RETRY_DELAY
        )

    def _tariff_option(self) -> str | None:
        """Return OPTARIF (historique) or NGTF (standard), if known."""
        return _decode_text(
            self.get(
                self.AttributeDefs.hist_tariff_option_or_std_supplier_price_schedule_name.id
            )
        )

    def is_single_index_contract(self) -> bool:
        """Return whether the contract has a single index (BASE), if known.

        Only contracts with several indexes have a day profile driving the
        virtual contacts. In historique mode, OPTARIF is an enumerated value.
        In standard mode, NGTF is free text chosen by the supplier, so the data
        the meter sends is used instead: the next day profile only switches to
        index 1 and the current index is 1.
        """
        mode = self.meter_mode()
        if mode is None:
            return False
        if not mode & LINKY_MODE_STANDARD:
            option = self._tariff_option()
            return option is not None and option.startswith("BASE")
        indexes = _profile_indexes(
            self.get(self.AttributeDefs.std_supplier_calendar_next_day_profile.id)
        )
        current = self.get(self.AttributeDefs.std_current_tariff_index_number.id)
        return indexes is not None and indexes <= {1} and current in (None, 1)

    def _irrelevant_attribute_ids(self) -> set[int]:
        """Return the attributes not relevant to the meter configuration.

        Nothing is hidden while the mode (or the contract) is unknown, except
        the day profile slots the meter did not send.
        """
        irrelevant = {
            profile_slot_key(attr_id, slot)
            for attr_id in _DAY_PROFILE_PREFIXES
            for slot in _PROFILE_SLOTS
            if slot > len(_day_profile_slots(self.get(attr_id)) or ())
        }
        mode = self.meter_mode()
        if mode is None:
            return irrelevant

        if mode & LINKY_MODE_THREE_PHASE:
            irrelevant |= _SINGLE_PHASE_ONLY_IDS
        else:
            irrelevant |= _THREE_PHASE_ONLY_IDS

        # Unknown (None) and 0 are alike, see `_TIERS`
        irrelevant |= {
            tier_key(tier)
            for tier in _TIERS
            if not self.endpoint.smartenergy_metering.get(
                getattr(Metering.AttributeDefs, f"current_tier{tier}_summ_delivered").id
            )
        }

        if mode & LINKY_MODE_STANDARD:
            irrelevant |= _HISTORIQUE_ONLY_IDS
            if not mode & LINKY_MODE_PRODUCER:
                irrelevant |= _PRODUCER_ONLY_IDS
            # Tempo contracts always announce a color for the current day
            register = _parse_status_register(
                self.get(self.AttributeDefs.std_status_register.id)
            )
            if register is not None and not register >> 24 & 0x03:
                irrelevant |= {
                    self.AttributeDefs.tempo_color_today.id,
                    self.AttributeDefs.tempo_color_tomorrow.id,
                }
            if self.is_single_index_contract():
                irrelevant.add(SINGLE_INDEX_KEY)
        else:
            irrelevant |= _STANDARD_ONLY_IDS
            option = self._tariff_option()
            if option is not None:
                if not option.startswith("BBR"):  # Tempo
                    irrelevant.add(self.AttributeDefs.hist_tomorrow_color.id)
                if not option.startswith("EJP"):
                    irrelevant.add(self.AttributeDefs.hist_ejp_start_notice.id)

        return irrelevant

    def refresh_non_reportable(self) -> None:
        """Re-read the non-reportable attributes whose refresh interval elapsed.

        The firmware never reports these attributes, so without this their
        value would only be read once when the device is configured.
        Reads triggered here update attributes again, but the groups are not
        due anymore, so this does not loop. A group whose read fails is due
        again after `_RETRY_DELAY`.
        """
        mode = self.meter_mode()
        now = time.monotonic()
        irrelevant = self._irrelevant_attribute_ids()
        chunks: list[tuple[int, tuple[str, ...]]] = []

        if self._last_config_check is None:
            self._last_config_check = now
        elif now - self._last_config_check >= _CONFIG_CHECK_INTERVAL:
            self._last_config_check = now
            self._check_meter_configuration()

        for index, group in enumerate(_REFRESH_GROUPS):
            if not group.applies(mode):
                continue
            last = self._last_refresh.get(index)
            if last is not None and now - last < group.interval:
                continue
            self._last_refresh[index] = now
            for chunk in group.reads:
                if names := tuple(
                    name for name in chunk if not self._is_hidden(name, irrelevant)
                ):
                    chunks.append((index, names))

        # Once the attributes of the new mode are read
        check_configuration, self._mode_changed = self._mode_changed, False
        if chunks:
            self.create_catching_task(
                self._read_chunks(chunks, check_configuration=check_configuration)
            )
        elif check_configuration:
            self._check_meter_configuration()

    async def _read_chunks(
        self,
        chunks: list[tuple[int, tuple[str, ...]]],
        *,
        check_configuration: bool = False,
    ) -> None:
        """Read attributes chunk by chunk, keeping responses within one frame."""
        for index, chunk in chunks:
            try:
                await self.read_attributes(list(chunk), allow_cache=False)
            except (TimeoutError, ZigbeeException) as exc:
                self.debug("Failed to refresh %s: %r", chunk, exc)
                interval = _REFRESH_GROUPS[index].interval
                self._last_refresh[index] = (
                    time.monotonic() - interval + min(interval, _RETRY_DELAY)
                )
        if check_configuration:
            self._check_meter_configuration()


class ZLinkyTICMetering(CustomCluster, Metering):
    """Override the SmartEnergy metering cluster for ZLinky_TIC.

    Reports ``current_summ_delivered`` (and tier indexes) in Wh but Home
    Assistant expects kWh: force ``multiplier=1, divisor=1000``.
    """

    _CONSTANT_ATTRIBUTES = {
        Metering.AttributeDefs.multiplier.id: 1,
        Metering.AttributeDefs.divisor.id: 1000,
    }

    def _update_attribute(self, attrid: Any, value: Any) -> None:
        """Update attribute, then refresh the manufacturer cluster if due.

        Metering is reported in both TIC modes, so it serves as a heartbeat.
        """
        super()._update_attribute(attrid, value)
        self.endpoint.zlinky_manufacturer_specific.refresh_non_reportable()

    def is_attribute_unsupported(self, attr: Any) -> bool:
        """Report unused indexes and injected energy of consumers as unsupported."""
        attr_id = _attribute_id(self, attr)
        for tier in _TIERS:
            if (
                attr_id
                == getattr(
                    Metering.AttributeDefs, f"current_tier{tier}_summ_delivered"
                ).id
                and tier_key(tier)
                in self.endpoint.zlinky_manufacturer_specific._irrelevant_attribute_ids()
            ):
                return True
        if attr_id == Metering.AttributeDefs.current_summ_received.id:
            mode = _linky_mode(self.endpoint)
            if mode is not None and not mode & LINKY_MODE_PRODUCER:
                return True
        return super().is_attribute_unsupported(attr)


class ZLinkyTICElectricalMeasurement(CustomCluster, ElectricalMeasurement):
    """Electrical measurement cluster hiding phases 2 and 3 on single-phase meters."""

    def is_attribute_unsupported(self, attr: Any) -> bool:
        """Report the phase 2 and 3 measurements as unsupported on single-phase meters."""
        if _attribute_id(self, attr) in _EM_THREE_PHASE_ONLY_IDS:
            mode = _linky_mode(self.endpoint)
            if mode is not None and not mode & LINKY_MODE_THREE_PHASE:
                return True
        return super().is_attribute_unsupported(attr)


# Entities of previous versions of the quirk: (platform, unique_id suffix)
_RETIRED_ENTITIES: Final = (
    *((Platform.BINARY_SENSOR, f"std_relay_{number}") for number in range(2, 9)),
    (Platform.SENSOR, "std_message_ultra_short"),
)


_HISTORIQUE_MODE_ALERT: Final = DeviceAlertMetadata(
    level=DeviceAlertLevel.WARNING,
    message=(
        "The TIC (customer tele-information) of this Linky meter is in historique"
        " mode, which only supports the legacy contracts (Base, Heures"
        " Creuses, EJP and Tempo) and provides less data: no voltage, status"
        " register, production, load curve or distributor indexes, and a"
        " coarser apparent power. Ask your electricity supplier to switch the"
        " TIC to standard mode (Enedis service F185). It is free and usually"
        " done within 24 working hours, and the ZLinky entities are updated"
        " automatically afterwards."
    ),
)


class ZLinkyTICDevice(QuirkV2Device):
    """ZHA device only exposing the entities relevant to the meter.

    The clusters report irrelevant attributes as unsupported, which ZHA ignores
    for quirk entities and binary sensors, so they are filtered out here. The
    entities are recomputed when the meter configuration changes. Hidden entities
    are only unloaded, their registry entries are kept so customizations and
    references survive (the meter state is inferred from cached values, which
    can be stale or incomplete). Only the registry entries of entities no longer
    declared by the quirk are deleted at startup, see `_RETIRED_ENTITIES`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init."""
        # Must exist before the parent init, which discovers the entities
        self._hidden_entity_keys: set[tuple[Platform, str]] = set()
        self._registry_cleaned = False
        super().__init__(*args, **kwargs)

    def _init_from_zigpy_device(self, zigpy_device: Any) -> None:
        """Initialize from the zigpy device, listening for configuration changes."""
        super()._init_from_zigpy_device(zigpy_device)

        # Callbacks are cleared above and run on teardown
        cluster = zigpy_device.endpoints[1].zlinky_manufacturer_specific
        self._on_remove_callbacks.append(
            cluster.on_event(ENTITIES_CHANGED_EVENT, self._handle_entities_changed)
        )

    @property
    def device_alerts(self) -> tuple[DeviceAlertMetadata, ...]:
        """Return the device alerts, warning while the TIC is in historique mode.

        A property rather than the cached one of the parent: the TIC mode can
        change at any time.
        """
        alerts = self._quirk_definition.device_alerts
        cluster = self._zigpy_device.endpoints[1].zlinky_manufacturer_specific
        if _is_historique(cluster.meter_mode()):
            alerts = (*alerts, _HISTORIQUE_MODE_ALERT)
        return alerts

    def _handle_entities_changed(self, _data: Any) -> None:
        self.gateway.async_create_task(
            self._update_entities(), name=f"recompute_entities_{self.ieee}"
        )

    async def _update_entities(self) -> None:
        """Recompute the entities, retrying soon on failure."""
        try:
            await self.recompute_entities()
        except Exception:  # noqa: BLE001
            self.error("Failed to update the entities, retrying later", exc_info=True)
            self._zigpy_device.endpoints[
                1
            ].zlinky_manufacturer_specific.retry_entities_update()

    @staticmethod
    def _is_hidden(entity: BaseEntity) -> bool:
        """Return whether the entity is irrelevant to the meter."""
        cluster = getattr(entity, "_cluster", None)
        attribute = getattr(entity, "_attribute_name", None)
        if not isinstance(cluster, _ZLINKY_CLUSTERS):
            return False
        # Derived from the status register, which every meter reports
        if entity.unique_id.endswith(_PRODUCER_ONLY_ENTITY_SUFFIXES):
            mode = _linky_mode(cluster.endpoint)
            if mode is not None and not mode & LINKY_MODE_PRODUCER:
                return True
        # Same attribute as the dry contact, which is always relevant
        if (
            _VIRTUAL_CONTACT_PREFIX in entity.unique_id
            and cluster.endpoint.zlinky_manufacturer_specific.is_single_index_contract()
        ):
            return True
        # Same attribute as the other slots of the day profile
        for marker, key in _PROFILE_SLOT_ENTITY_KEYS.items():
            if marker in entity.unique_id:
                if (
                    key
                    in cluster.endpoint.zlinky_manufacturer_specific._irrelevant_attribute_ids()
                ):
                    return True
                break
        return (
            attribute in cluster.attributes_by_name
            and cluster.is_attribute_unsupported(attribute)
        )

    def discover_entities(self) -> Iterator[BaseEntity]:
        """Yield the entities, except the ones irrelevant to the meter."""
        self._hidden_entity_keys = set()
        for entity in super().discover_entities():
            if self._is_hidden(entity):
                self._hidden_entity_keys.add((entity.PLATFORM, entity.unique_id))
            else:
                yield entity

    async def recompute_entities(self) -> None:
        """Recompute the entities, unloading the ones now irrelevant to the meter."""
        for entity in list(self._platform_entities.values()):
            if self._is_hidden(entity):
                self.debug("Removing irrelevant entity %s", entity)
                await self._remove_entity(entity, remove=False)
        await super().recompute_entities()

    async def async_initialize(self, from_cache: bool = False) -> None:
        """Initialize, then delete the registry entries of retired entities."""
        # At startup, Home Assistant listens once the device is initialized
        listened = self._initialized
        await super().async_initialize(from_cache)
        if not listened or self._registry_cleaned:
            return
        self._registry_cleaned = True
        retired = {
            (platform, f"{self.ieee}-1-{suffix}")
            for platform, suffix in _RETIRED_ENTITIES
        }
        for platform, unique_id in retired:
            self.emit(
                DeviceEntityRemovedEvent.event_type,
                DeviceEntityRemovedEvent(
                    platform=platform, unique_id=unique_id, remove=True
                ),
            )


def _linky_mode(endpoint: Any) -> int | None:
    """Return the TIC mode known to the manufacturer cluster of `endpoint`."""
    return endpoint.zlinky_manufacturer_specific.meter_mode()


def _attribute_id(cluster: Any, attr: Any) -> int | None:
    """Return the id of attribute `attr` (id or name), None if not defined."""
    try:
        return cluster.find_attribute(attr).id
    except KeyError:
        return None


def _is_standard(mode: int | None) -> bool:
    return mode is not None and bool(mode & LINKY_MODE_STANDARD)


def _is_historique(mode: int | None) -> bool:
    return mode is not None and not mode & LINKY_MODE_STANDARD


_ZLINKY_CLUSTERS = (
    ZLinkyTICManufacturerCluster,
    ZLinkyTICMetering,
    ZLinkyTICElectricalMeasurement,
)


class _RefreshGroup(NamedTuple):
    """Non-reportable attributes re-read together at a fixed interval (seconds)."""

    interval: float
    applies: Callable[[int | None], bool]
    reads: tuple[tuple[str, ...], ...]


_ATTRS = ZLinkyTICManufacturerCluster.AttributeDefs

_VIRTUAL_ATTRIBUTE_IDS: Final = {
    _ATTRS.tempo_color_today.id,
    _ATTRS.tempo_color_tomorrow.id,
}
_STANDARD_ONLY_IDS: Final = _VIRTUAL_ATTRIBUTE_IDS | {
    attr.id
    for attr in ZLinkyTICManufacturerCluster.attributes.values()
    if attr.name.startswith("std_")
}
# OPTARIF (historique) and NGTF (standard) share attribute 0x0000
_HISTORIQUE_ONLY_IDS: Final = {_ATTRS.linky_status.id} | {
    attr.id
    for attr in ZLinkyTICManufacturerCluster.attributes.values()
    if attr.name.startswith("hist_")
    and attr != _ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name
}
_THREE_PHASE_ONLY_IDS: Final = {
    _ATTRS.hist_potentials_presence.id,
    _ATTRS.hist_current_exceeding_warning_phase_1.id,
    _ATTRS.hist_current_exceeding_warning_phase_2.id,
    _ATTRS.hist_current_exceeding_warning_phase_3.id,
    _ATTRS.std_apparent_power_withdrawn_max_phase_2_n_1.id,
    _ATTRS.std_apparent_power_withdrawn_max_phase_3_n_1.id,
}
# Three-phase historique meters warn per phase instead (ADIR1-3)
_SINGLE_PHASE_ONLY_IDS: Final = {_ATTRS.hist_subscribed_power_exceeding_warning.id}
_PRODUCER_ONLY_IDS: Final = {
    _ATTRS.std_apparent_power_injected_instantaneous.id,
    _ATTRS.std_apparent_power_injected_max.id,
    _ATTRS.std_apparent_power_injected_max_1.id,
    _ATTRS.std_injected_active_load_curve_point_n.id,
    _ATTRS.std_injected_active_load_curve_point_n_1.id,
}
# Entities of producer meters whose attribute is also reported by consumer ones
_PRODUCER_ONLY_ENTITY_SUFFIXES: Final = ("-injecting",)
_VIRTUAL_CONTACT_PREFIX: Final = "-virtual_contact_"
# Unique id prefix of the day profile slot entities, by profile attribute id
_DAY_PROFILE_PREFIXES: Final = {
    ZLinkyTICManufacturerCluster.AttributeDefs.std_supplier_calendar_next_day_profile.id: "pjourf_plus_1",
    ZLinkyTICManufacturerCluster.AttributeDefs.std_next_peak_day_profile.id: "ppointe1",
}
# Unique id marker of the entities of each day profile slot, to its pseudo id
_PROFILE_SLOT_ENTITY_KEYS: Final = {
    f"-{prefix}_slot_{slot}_": profile_slot_key(attr_id, slot)
    for attr_id, prefix in _DAY_PROFILE_PREFIXES.items()
    for slot in _PROFILE_SLOTS
}
# `active_power_ph_b` is not phase 2 but CCASN-1 on every meter, so it is kept
_EM_ATTRS = ElectricalMeasurement.AttributeDefs
_EM_THREE_PHASE_ONLY_IDS: Final = {
    _EM_ATTRS.rms_voltage_ph_b.id,
    _EM_ATTRS.rms_voltage_ph_c.id,
    _EM_ATTRS.rms_current_ph_b.id,
    _EM_ATTRS.rms_current_ph_c.id,
    _EM_ATTRS.rms_current_max_ph_b.id,
    _EM_ATTRS.rms_current_max_ph_c.id,
    _EM_ATTRS.active_power_ph_c.id,
    _EM_ATTRS.active_power_max_ph_b.id,
    _EM_ATTRS.active_power_max_ph_c.id,
    _EM_ATTRS.apparent_power_ph_b.id,
    _EM_ATTRS.apparent_power_ph_c.id,
    _EM_ATTRS.average_rms_voltage_measure_period_ph_b.id,
    _EM_ATTRS.average_rms_voltage_meas_period_ph_c.id,
}

# Hidden attributes are skipped when refreshing, see `is_attribute_unsupported`
_REFRESH_GROUPS: Final = (
    _RefreshGroup(
        interval=60,
        applies=lambda mode: True,
        reads=((_ATTRS.linky_mode.name, _ATTRS.linky_tariff_period.name),),
    ),
    # OPTARIF (historique) or NGTF (standard), also needed to trust mode 0
    _RefreshGroup(
        interval=3600,
        applies=lambda mode: True,
        reads=((_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name.name,),),
    ),
    _RefreshGroup(
        interval=60,
        applies=_is_standard,
        reads=(
            (
                _ATTRS.std_current_date_and_time.name,
                _ATTRS.std_status_register.name,
                _ATTRS.std_relay.name,
            ),
            (
                _ATTRS.std_current_supplier_price_description.name,
                _ATTRS.std_current_tariff_index_number.name,
            ),
        ),
    ),
    _RefreshGroup(
        interval=60,
        applies=_is_historique,
        reads=((_ATTRS.linky_status.name,),),
    ),
    # Reportable, but reporting is only configured for the entities existing
    # when ZHA configures the device (not when hidden then, e.g. in historique
    # mode) and the firmware report table is limited: read them at the
    # maximum reporting interval of energy counters as well
    _RefreshGroup(
        interval=_ENERGY_REPORTING_MAX_INTERVAL,
        applies=_is_standard,
        reads=(
            (
                _ATTRS.std_active_energy_withdrawn_distributor_index_01.name,
                _ATTRS.std_active_energy_withdrawn_distributor_index_02.name,
                _ATTRS.std_active_energy_withdrawn_distributor_index_03.name,
                _ATTRS.std_active_energy_withdrawn_distributor_index_04.name,
            ),
        ),
    ),
    _RefreshGroup(
        interval=3600,
        applies=_is_standard,
        reads=(
            (
                _ATTRS.std_apparent_power_withdrawn_max_phase_1_n_1.name,
                _ATTRS.std_apparent_power_withdrawn_max_phase_2_n_1.name,
                _ATTRS.std_apparent_power_withdrawn_max_phase_3_n_1.name,
            ),
            (
                _ATTRS.std_supplier_calendar_current_day_number.name,
                _ATTRS.std_supplier_calendar_next_day_number.name,
            ),
            (
                _ATTRS.std_mobile_peak_start_1.name,
                _ATTRS.std_mobile_peak_end_1.name,
                _ATTRS.std_mobile_peak_start_2.name,
                _ATTRS.std_mobile_peak_end_2.name,
                _ATTRS.std_mobile_peak_start_3.name,
                _ATTRS.std_mobile_peak_end_3.name,
            ),
            (_ATTRS.std_message_short.name,),
            (_ATTRS.std_supplier_calendar_next_day_profile.name,),
            (_ATTRS.std_next_peak_day_profile.name,),
            (
                _ATTRS.std_apparent_power_injected_max_1.name,
                _ATTRS.std_injected_active_load_curve_point_n_1.name,
            ),
        ),
    ),
    _RefreshGroup(
        interval=3600,
        applies=_is_historique,
        reads=(
            (
                _ATTRS.hist_schedule_peak_hours_off_peak_hours.name,
                _ATTRS.hist_potentials_presence.name,
            ),
        ),
    ),
)


def _decode_text(raw: object) -> str | None:
    """Decode a TIC string, dropping the padding spaces Enedis frames carry.

    The firmware pads its buffers with NULs, possibly followed by stale bytes
    of a longer previous value, so the string ends at the first NUL.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("ascii", errors="ignore")
    if not isinstance(raw, str):
        return None
    raw = raw.split("\x00", 1)[0]
    return " ".join(raw.split()) or None


# MSG1 when the supplier has nothing to say
_NO_MESSAGE: Final = "PAS DE MESSAGE"


def _parse_message(raw: object) -> str:
    """Decode a supplier message, empty when there is none."""
    text = _decode_text(raw)
    if text is None or text == _NO_MESSAGE:
        return ""
    return text


# Season character of a TIC timestamp: summer (E) or winter (H) time in France.
# Lowercase means the meter clock is running in degraded mode.
_SEASON_UTC_OFFSET: Final = {"E": timedelta(hours=2), "H": timedelta(hours=1)}


def _parse_linky_date(raw: object) -> datetime | None:
    """Parse a TIC timestamp (``SAAMMJJhhmmss``, e.g. ``E260701015506``)."""
    text = _decode_text(raw)
    if text is None or len(text) != 13:
        return None
    offset = _SEASON_UTC_OFFSET.get(text[0].upper())
    if offset is None:
        return None
    try:
        return datetime(
            2000 + int(text[1:3]),
            int(text[3:5]),
            int(text[5:7]),
            int(text[7:9]),
            int(text[9:11]),
            int(text[11:13]),
            tzinfo=timezone(offset),
        )
    except ValueError:
        return None


_STATUS_REGISTER_PATTERN: Final = re.compile(r"[0-9A-Fa-f]{8}")


def _parse_status_register(raw: object) -> int | None:
    """Parse the STGE status register, sent as 8 hexadecimal characters."""
    text = _decode_text(raw)
    # `int(text, 16)` alone would also accept signs, `0x` and underscores
    if text is None or not _STATUS_REGISTER_PATTERN.fullmatch(text):
        return None
    return int(text, 16)


def _status_field(
    shift: int, width: int, labels: tuple[Any, ...]
) -> Callable[[object], Any]:
    """Return a converter mapping a status register bit field to a label."""

    def convert(raw: object) -> Any:
        register = _parse_status_register(raw)
        if register is None:
            return None
        value = (register >> shift) & ((1 << width) - 1)
        return labels[value] if value < len(labels) else None

    return convert


def _status_flag(bit: int) -> Callable[[object], bool | None]:
    """Return a converter extracting a single status register bit."""
    return _status_field(bit, 1, (False, True))


_CUT_OFF_SWITCH_STATES: Final = (
    "closed",
    "open_overpower",
    "open_overvoltage",
    "open_load_shedding",
    "open_remote_order",
    "open_overheating_high_current",
    "open_overheating_low_current",
)
_cut_off_switch_state = _status_field(1, 3, tuple(range(8)))


def _cut_off_switch_open(raw: object) -> bool | None:
    """Return whether the cut-off switch is open, whatever the reason."""
    state = _cut_off_switch_state(raw)
    return None if state is None else state != 0


_CPL_STATUSES: Final = ("new_unlock", "new_lock", "registered")
_MOBILE_PEAKS: Final = ("none", "pm1", "pm2", "pm3")


def _decode_schedule_letter(value: int) -> str | None:
    """Decode HHPHC, a schedule letter (e.g. ``A``) in a uint8, as its ASCII code.

    Unconfirmed: the firmware may never store it, and then reports 0 (None).
    """
    char = chr(value)
    return char if char.isascii() and char.isalpha() else None


def _relay_closed(number: int) -> Callable[[int], bool]:
    """Return a converter telling whether relay `number` (1-8) is closed."""
    return lambda value: bool(value & (1 << (number - 1)))


_DRY_CONTACT_ACTIONS: Final = {0b10: "open", 0b11: "closed"}


# Maximum length of a Home Assistant state
_MAX_STATE_LENGTH: Final = 255
_PROFILE_BLOCK_PATTERN: Final = re.compile(r"[0-9]{4}[0-9A-Fa-f]{4}")


def _profile_blocks(raw: object) -> list[tuple[int, int, int]] | None:
    """Return the valid ``(hour, minute, action)`` blocks of a day profile.

    See `_parse_day_profile` for the format. None when no profile is known.
    """
    text = _decode_text(raw)
    if text is None:
        return None
    blocks = []
    for block in text.split():
        if not _PROFILE_BLOCK_PATTERN.fullmatch(block):
            continue
        hour, minute = int(block[0:2]), int(block[2:4])
        if hour < 24 and minute < 60:
            blocks.append((hour, minute, int(block[4:], 16)))
    return blocks or None


def _profile_indexes(raw: object) -> set[int] | None:
    """Return the supplier tariff indexes a day profile switches to, if known."""
    blocks = _profile_blocks(raw)
    if blocks is None:
        return None
    return {action & 0x0F for _, _, action in blocks} & set(_TIERS)


def _parse_day_profile(raw: object) -> str | None:
    r"""Summarize a PJOURF+1 / PPOINTE1 day profile as tariff index changes.

    Output: slots joined by ``; ``, each ``HH:MM`` followed by a space and
    parts joined by ``, ``: ``index N``, ``dry contact open|closed`` and
    ``contacts A+B`` (the virtual contacts set from that time), or
    ``contacts none`` when the virtual contacts of the previous slot are all
    cleared, e.g.
    ``00:00 index 1, dry contact closed; 06:00 index 2, dry contact open``.
    Slots changing nothing are omitted, as are the last slots if needed to fit
    a Home Assistant state. To parse, split on ``; `` for the slots, on the
    first space for the time, then on ``, `` for the parts.

    The Enedis profile is made of up to 11 ``HHMMSSSS`` blocks, but the
    firmware only sends the first 8. Each is a decimal ``HHMM`` start time and
    a hex action whose 4 low bits are the supplier tariff index (1-10) to switch to (0 and
    11-15: unchanged) and whose bits 4-10 are the state of the 7 virtual
    contacts, bits 15-14 being the dry contact (``10`` opens it, ``11`` closes
    it, ``00`` leaves it, ``01`` follows the Tempo configuration, which is not
    reported). Unused blocks are ``NONUTILE``.
    """
    blocks = _profile_blocks(raw)
    if blocks is None:
        return None
    changes: list[str] = []
    length = 0
    previous_contacts: list[str] = []
    for hour, minute, action in blocks:
        index = action & 0x0F
        contacts = [
            str(number) for number in range(1, 8) if action & (1 << (number + 3))
        ]
        change = []
        if 1 <= index <= 10:
            change.append(f"index {index}")
        if dry_contact := _DRY_CONTACT_ACTIONS.get(action >> 14 & 0x03):
            change.append(f"dry contact {dry_contact}")
        if contacts:
            change.append(f"contacts {'+'.join(contacts)}")
        elif previous_contacts:
            change.append("contacts none")
        previous_contacts = contacts
        if not change:
            continue
        slot = f"{hour:02}:{minute:02} {', '.join(change)}"
        length += len(slot) + (2 if changes else 0)
        if length > _MAX_STATE_LENGTH:
            break
        changes.append(slot)
    return "; ".join(changes) or None


def _day_profile_slots(raw: object) -> list[tuple[str, int | None, str]] | None:
    """Return the ``(start, index, contacts)`` slots of a day profile.

    Each slot is the state from its ``HH:MM`` start time: the supplier tariff
    index (None until the profile sets one) and the closed contacts as a comma
    separated list, ``0`` being the dry contact and 1-7 the virtual contacts,
    e.g. ``0,1,3`` (empty when all are open). An action leaving the index or
    the dry contact unchanged keeps the state of the previous slot, and slots
    changing nothing are omitted. See `_parse_day_profile` for the format.
    """
    blocks = _profile_blocks(raw)
    if blocks is None:
        return None
    slots = []
    index: int | None = None
    dry_contact_closed = False
    previous: tuple[int | None, str] = (None, "")
    for hour, minute, action in blocks:
        if 1 <= action & 0x0F <= 10:
            index = action & 0x0F
        if dry_contact := _DRY_CONTACT_ACTIONS.get(action >> 14 & 0x03):
            dry_contact_closed = dry_contact == "closed"
        contacts = ["0"] if dry_contact_closed else []
        contacts += [
            str(number) for number in range(1, 8) if action & (1 << (number + 3))
        ]
        state = (index, ",".join(contacts))
        if state == previous:
            continue
        previous = state
        slots.append((f"{hour:02}:{minute:02}", *state))
    return slots or None


# Fields of `_day_profile_slots`: (unique id suffix, fallback name)
_PROFILE_SLOT_FIELDS: Final = (
    ("start", "start"),
    ("index", "tariff index"),
    ("contacts", "contacts"),
)


def _profile_slot_field(slot: int, field: int) -> Callable[[object], Any]:
    """Return a converter extracting field `field` of slot `slot` (1-8)."""

    def convert(raw: object) -> Any:
        slots = _day_profile_slots(raw)
        if slots is None or slot > len(slots):
            return None
        return slots[slot - 1][field]

    return convert


_ENERGY_REPORTING = ReportingConfig(
    min_interval=60,
    max_interval=_ENERGY_REPORTING_MAX_INTERVAL,
    reportable_change=1,
)


def _apply_lixee_entities(builder: QuirkBuilder) -> QuirkBuilder:
    """Add the ZLinky entities.

    Entities irrelevant to the TIC mode, phases, production or contract are
    not created, see `ZLinkyTICManufacturerCluster.is_attribute_unsupported`.
    Mobile peak contracts cannot be detected, so their entities are disabled by
    default, as are debug values.
    """
    a = ZLinkyTICManufacturerCluster.AttributeDefs
    m = ZLinkyTICManufacturerCluster.cluster_id
    builder = (
        builder.zha_device_class(ZLinkyTICDevice)
        .replaces(ZLinkyTICMetering)
        .replaces(ZLinkyTICElectricalMeasurement)
        .replaces(ZLinkyTICManufacturerCluster)
        # The ZLinky reports CCASN-1 there, not the phase 2 active power
        .change_entity_metadata(
            endpoint_id=1,
            cluster_id=ElectricalMeasurement.cluster_id,
            unique_id_suffix=ElectricalMeasurement.AttributeDefs.active_power_ph_b.name,
            new_translation_key="ccasn_1",
            new_fallback_name="Withdrawn active load curve point (n-1)",
        )
    )

    # === Active energy withdrawn, distributor indexes (EASD01..04) ===
    # Grid counters: the supplier ones, used for billing, are the metering tiers
    for index, attr in enumerate(
        (
            a.std_active_energy_withdrawn_distributor_index_01,
            a.std_active_energy_withdrawn_distributor_index_02,
            a.std_active_energy_withdrawn_distributor_index_03,
            a.std_active_energy_withdrawn_distributor_index_04,
        ),
        start=1,
    ):
        builder = builder.sensor(
            attribute_name=attr.name,
            cluster_id=m,
            divisor=1000,
            device_class=SensorDeviceClass.ENERGY,
            state_class=SensorStateClass.TOTAL_INCREASING,
            unit=UnitOfEnergy.KILO_WATT_HOUR,
            reporting_config=_ENERGY_REPORTING,
            initially_disabled=True,
            translation_key=f"easd{index:02}",
            fallback_name=f"EASD{index:02} (index {index:02})",
        )

    # === Active energy withdrawn, supplier indexes 7-10 (EASF07..10) ===
    # ZHA creates the first six tiers itself, from the metering cluster. Like
    # these, 7-10 only exist once the meter reports energy on them, see `_TIERS`.
    for index in range(7, 11):
        builder = builder.sensor(
            attribute_name=f"current_tier{index}_summ_delivered",
            cluster_id=Metering.cluster_id,
            divisor=1000,
            device_class=SensorDeviceClass.ENERGY,
            state_class=SensorStateClass.TOTAL_INCREASING,
            unit=UnitOfEnergy.KILO_WATT_HOUR,
            reporting_config=_ENERGY_REPORTING,
            translation_key=f"tier{index}_summation_delivered",
            fallback_name=f"Tier {index} summation delivered",
        )

    # === Production (standard mode, producer meters only) ===
    builder = (
        builder.sensor(
            attribute_name=a.std_apparent_power_injected_instantaneous.name,
            cluster_id=m,
            device_class=SensorDeviceClass.APPARENT_POWER,
            state_class=SensorStateClass.MEASUREMENT,
            unit=UnitOfApparentPower.VOLT_AMPERE,
            translation_key="sinsti",
            fallback_name="Instantaneous apparent power injected",
        )
        .sensor(
            attribute_name=a.std_apparent_power_injected_max.name,
            cluster_id=m,
            device_class=SensorDeviceClass.APPARENT_POWER,
            state_class=SensorStateClass.MEASUREMENT,
            unit=UnitOfApparentPower.VOLT_AMPERE,
            translation_key="smaxin",
            fallback_name="Apparent power injected max (today)",
        )
        .sensor(
            attribute_name=a.std_apparent_power_injected_max_1.name,
            cluster_id=m,
            device_class=SensorDeviceClass.APPARENT_POWER,
            state_class=SensorStateClass.MEASUREMENT,
            unit=UnitOfApparentPower.VOLT_AMPERE,
            translation_key="smaxin_1",
            fallback_name="Apparent power injected max (yesterday)",
        )
        # Load curve points are average active powers over the last periods
        .sensor(
            attribute_name=a.std_injected_active_load_curve_point_n.name,
            cluster_id=m,
            device_class=SensorDeviceClass.POWER,
            state_class=SensorStateClass.MEASUREMENT,
            unit=UnitOfPower.WATT,
            translation_key="ccain",
            fallback_name="Injected active load curve point (n)",
        )
        .sensor(
            attribute_name=a.std_injected_active_load_curve_point_n_1.name,
            cluster_id=m,
            device_class=SensorDeviceClass.POWER,
            state_class=SensorStateClass.MEASUREMENT,
            unit=UnitOfPower.WATT,
            translation_key="ccain_1",
            fallback_name="Injected active load curve point (n-1)",
        )
    )

    # === Maximum apparent power withdrawn yesterday (phase 1 = single-phase) ===
    for phase, attr in enumerate(
        (
            a.std_apparent_power_withdrawn_max_phase_1_n_1,
            a.std_apparent_power_withdrawn_max_phase_2_n_1,
            a.std_apparent_power_withdrawn_max_phase_3_n_1,
        ),
        start=1,
    ):
        builder = builder.sensor(
            attribute_name=attr.name,
            cluster_id=m,
            device_class=SensorDeviceClass.APPARENT_POWER,
            state_class=SensorStateClass.MEASUREMENT,
            unit=UnitOfApparentPower.VOLT_AMPERE,
            translation_key=f"smaxsn_phase_{phase}_n_1",
            fallback_name=f"Apparent power withdrawn max phase {phase} (yesterday)",
        )

    # === Tariff ===
    builder = (
        builder.sensor(
            attribute_name=a.linky_tariff_period.name,
            cluster_id=m,
            attribute_converter=_decode_text,
            translation_key="linky_tariff_period",
            fallback_name="Linky tariff period",
        )
        .sensor(
            attribute_name=a.hist_tariff_option_or_std_supplier_price_schedule_name.name,
            cluster_id=m,
            attribute_converter=_decode_text,
            initially_disabled=True,
            translation_key="optarif_ngtf",
            fallback_name="Tariff option or supplier schedule",
        )
        .sensor(
            attribute_name=a.std_current_supplier_price_description.name,
            cluster_id=m,
            attribute_converter=_decode_text,
            initially_disabled=True,
            translation_key="ltarf",
            fallback_name="Current supplier price label",
        )
        .sensor(
            attribute_name=a.std_current_tariff_index_number.name,
            cluster_id=m,
            initially_disabled=True,
            translation_key="ntarf",
            fallback_name="Current tariff index number",
        )
        .sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_status_field(14, 2, (1, 2, 3, 4)),
            initially_disabled=True,
            unique_id_suffix="distributor_tariff_index",
            translation_key="distributor_tariff_index",
            fallback_name="Current distributor tariff index",
        )
        .sensor(
            attribute_name=a.std_supplier_calendar_next_day_profile.name,
            cluster_id=m,
            attribute_converter=_parse_day_profile,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="pjourf_plus_1",
            fallback_name="Supplier calendar next day profile",
        )
        .sensor(
            attribute_name=a.std_supplier_calendar_current_day_number.name,
            cluster_id=m,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="njourf",
            fallback_name="Supplier calendar current day number",
        )
        .sensor(
            attribute_name=a.std_supplier_calendar_next_day_number.name,
            cluster_id=m,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="njourf_plus_1",
            fallback_name="Supplier calendar next day number",
        )
    )

    # === Day profile slots, for automations ===
    # Only the slots the meter sent exist, see `profile_slot_key`
    for attribute, name in (
        (a.std_supplier_calendar_next_day_profile, "Supplier calendar next day"),
        (a.std_next_peak_day_profile, "Next peak day"),
    ):
        prefix = _DAY_PROFILE_PREFIXES[attribute.id]
        for slot in _PROFILE_SLOTS:
            for field, (suffix, label) in enumerate(_PROFILE_SLOT_FIELDS):
                builder = builder.sensor(
                    attribute_name=attribute.name,
                    cluster_id=m,
                    attribute_converter=_profile_slot_field(slot, field),
                    entity_type=EntityType.DIAGNOSTIC,
                    initially_disabled=True,
                    unique_id_suffix=f"{prefix}_slot_{slot}_{suffix}",
                    translation_key=f"{prefix}_slot_{slot}_{suffix}",
                    fallback_name=f"{name} slot {slot} {label}",
                )

    # === Tempo / EJP / mobile peaks ===
    builder = (
        builder.enum(
            attribute_name=a.tempo_color_today.name,
            enum_class=TempoColor,
            cluster_id=m,
            entity_platform=EntityPlatform.SENSOR,
            entity_type=EntityType.STANDARD,
            translation_key="tempo_color_today",
            fallback_name="Tempo color today",
        )
        .enum(
            attribute_name=a.tempo_color_tomorrow.name,
            enum_class=TempoColor,
            cluster_id=m,
            entity_platform=EntityPlatform.SENSOR,
            entity_type=EntityType.STANDARD,
            translation_key="tempo_color_tomorrow",
            fallback_name="Tempo color tomorrow",
        )
        .sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_status_field(28, 2, _MOBILE_PEAKS),
            initially_disabled=True,
            unique_id_suffix="mobile_peak_notice",
            translation_key="mobile_peak_notice",
            fallback_name="Mobile peak notice",
        )
        .sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_status_field(30, 2, _MOBILE_PEAKS),
            initially_disabled=True,
            unique_id_suffix="mobile_peak",
            translation_key="mobile_peak",
            fallback_name="Mobile peak",
        )
        .sensor(
            attribute_name=a.std_next_peak_day_profile.name,
            cluster_id=m,
            attribute_converter=_parse_day_profile,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="ppointe1",
            fallback_name="Next peak day profile",
        )
        .sensor(
            attribute_name=a.hist_tomorrow_color.name,
            cluster_id=m,
            attribute_converter=_decode_text,
            translation_key="demain",
            fallback_name="Tomorrow color (Tempo)",
        )
        # The firmware never clears PEJP, ADPS and ADIR1-3: the last notice or
        # warning is kept until the ZLinky restarts, so they are disabled
        .sensor(
            attribute_name=a.hist_ejp_start_notice.name,
            cluster_id=m,
            device_class=SensorDeviceClass.DURATION,
            state_class=SensorStateClass.MEASUREMENT,
            unit=UnitOfTime.MINUTES,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="pejp",
            fallback_name="EJP start notice",
        )
    )
    for name, attr in (
        ("dpm1", a.std_mobile_peak_start_1),
        ("fpm1", a.std_mobile_peak_end_1),
        ("dpm2", a.std_mobile_peak_start_2),
        ("fpm2", a.std_mobile_peak_end_2),
        ("dpm3", a.std_mobile_peak_start_3),
        ("fpm3", a.std_mobile_peak_end_3),
    ):
        number = name[-1]
        builder = builder.sensor(
            attribute_name=attr.name,
            cluster_id=m,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key=name,
            fallback_name=f"Mobile peak {number} {'start' if name[0] == 'd' else 'end'}",
        )

    # === Alarms and meter status (status register STGE) ===
    builder = (
        builder.binary_sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_status_flag(7),
            device_class=BinarySensorDeviceClass.PROBLEM,
            entity_type=EntityType.STANDARD,
            unique_id_suffix="reference_power_exceeded",
            translation_key="reference_power_exceeded",
            fallback_name="Reference power exceeded",
        )
        .binary_sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_status_flag(6),
            device_class=BinarySensorDeviceClass.PROBLEM,
            entity_type=EntityType.DIAGNOSTIC,
            unique_id_suffix="overvoltage",
            translation_key="overvoltage",
            fallback_name="Overvoltage",
        )
        .binary_sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_cut_off_switch_open,
            device_class=BinarySensorDeviceClass.OPENING,
            entity_type=EntityType.STANDARD,
            unique_id_suffix="cut_off_switch_open",
            translation_key="cut_off_switch_open",
            fallback_name="Cut-off switch open",
        )
        .sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_status_field(1, 3, _CUT_OFF_SWITCH_STATES),
            entity_type=EntityType.DIAGNOSTIC,
            unique_id_suffix="cut_off_switch",
            translation_key="cut_off_switch",
            fallback_name="Cut-off switch",
        )
        .binary_sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_status_flag(4),
            device_class=BinarySensorDeviceClass.OPENING,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            unique_id_suffix="terminal_cover",
            translation_key="terminal_cover",
            fallback_name="Terminal cover",
        )
        .binary_sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_status_flag(9),
            entity_type=EntityType.STANDARD,
            initially_disabled=True,
            unique_id_suffix="injecting",
            translation_key="injecting",
            fallback_name="Injecting energy",
        )
        .binary_sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_status_flag(16),
            device_class=BinarySensorDeviceClass.PROBLEM,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            unique_id_suffix="clock_degraded",
            translation_key="clock_degraded",
            fallback_name="Meter clock degraded",
        )
        .sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_status_field(21, 2, _CPL_STATUSES),
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            unique_id_suffix="cpl_status",
            translation_key="cpl_status",
            fallback_name="CPL status",
        )
        .binary_sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_status_flag(23),
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            unique_id_suffix="cpl_synchronized",
            translation_key="cpl_synchronized",
            fallback_name="CPL synchronized",
        )
        .sensor(
            attribute_name=a.std_status_register.name,
            cluster_id=m,
            attribute_converter=_decode_text,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="stge",
            fallback_name="Status register",
        )
        .sensor(
            attribute_name=a.hist_subscribed_power_exceeding_warning.name,
            cluster_id=m,
            device_class=SensorDeviceClass.CURRENT,
            state_class=SensorStateClass.MEASUREMENT,
            unit=UnitOfElectricCurrent.AMPERE,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="adps",
            fallback_name="Subscribed power exceeding warning",
        )
    )
    for phase, attr in enumerate(
        (
            a.hist_current_exceeding_warning_phase_1,
            a.hist_current_exceeding_warning_phase_2,
            a.hist_current_exceeding_warning_phase_3,
        ),
        start=1,
    ):
        builder = builder.sensor(
            attribute_name=attr.name,
            cluster_id=m,
            device_class=SensorDeviceClass.CURRENT,
            state_class=SensorStateClass.MEASUREMENT,
            unit=UnitOfElectricCurrent.AMPERE,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key=f"adir{phase}",
            fallback_name=f"Current exceeding warning phase {phase}",
        )

    # === Relays: relay 1 is the physical dry contact, 2-8 are virtual ===
    # The virtual contacts follow the actions of the day profile, so they only
    # exist for contracts with several indexes, see `SINGLE_INDEX_KEY`.
    builder = builder.binary_sensor(
        attribute_name=a.std_relay.name,
        cluster_id=m,
        attribute_converter=_relay_closed(1),
        entity_type=EntityType.STANDARD,
        translation_key="dry_contact",
        fallback_name="Dry contact",
    )
    for contact in range(1, 8):
        builder = builder.binary_sensor(
            attribute_name=a.std_relay.name,
            cluster_id=m,
            attribute_converter=_relay_closed(contact + 1),
            entity_type=EntityType.STANDARD,
            unique_id_suffix=f"virtual_contact_{contact}",
            translation_key=f"virtual_contact_{contact}",
            fallback_name=f"Virtual contact {contact}",
        )

    # === Message from the supplier (the ultra-short one is unused) ===
    builder = builder.sensor(
        attribute_name=a.std_message_short.name,
        cluster_id=m,
        attribute_converter=_parse_message,
        entity_type=EntityType.DIAGNOSTIC,
        initially_disabled=True,
        translation_key="msg1",
        fallback_name="Short message from operator",
    )

    # === TIC link diagnostics ===
    builder = (
        builder.enum(
            attribute_name=a.linky_mode.name,
            enum_class=LinkyMode,
            cluster_id=m,
            entity_platform=EntityPlatform.SENSOR,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="linky_mode",
            fallback_name="Linky mode",
        )
        # Historique mode only exposes a subset of the data: users should ask
        # their supplier to switch the TIC to standard mode.
        .binary_sensor(
            attribute_name=a.linky_mode.name,
            cluster_id=m,
            attribute_converter=lambda mode: not mode & LINKY_MODE_STANDARD,
            device_class=BinarySensorDeviceClass.PROBLEM,
            entity_type=EntityType.DIAGNOSTIC,
            unique_id_suffix="historique_mode",
            translation_key="historique_mode",
            fallback_name="TIC in historique mode",
        )
        .sensor(
            attribute_name=a.std_current_date_and_time.name,
            cluster_id=m,
            attribute_converter=_parse_linky_date,
            device_class=SensorDeviceClass.TIMESTAMP,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="linky_date",
            fallback_name="Linky current date and time",
        )
        .sensor(
            attribute_name=a.linky_status.name,
            cluster_id=m,
            attribute_converter=_decode_text,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="motdetat",
            fallback_name="Linky status",
        )
        .sensor(
            attribute_name=a.hist_schedule_peak_hours_off_peak_hours.name,
            cluster_id=m,
            attribute_converter=_decode_schedule_letter,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="hhphc",
            fallback_name="Peak / off-peak schedule",
        )
        .sensor(
            attribute_name=a.hist_potentials_presence.name,
            cluster_id=m,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="ppot",
            fallback_name="Potentials presence (tri-phase)",
        )
        .sensor(
            attribute_name=a.linky_acquisition_time.name,
            cluster_id=m,
            device_class=SensorDeviceClass.DURATION,
            state_class=SensorStateClass.MEASUREMENT,
            unit=UnitOfTime.SECONDS,
            entity_type=EntityType.DIAGNOSTIC,
            initially_disabled=True,
            translation_key="acquisition_time",
            fallback_name="Linky acquisition time",
        )
    )

    return builder


# Before v12: no PowerConfiguration, no Time, no Tuya cluster. Also used when
# the firmware version is unknown, so no cluster the device may lack is added.
(
    _apply_lixee_entities(
        QuirkBuilder(LIXEE, "ZLinky_TIC").firmware_version_filter(
            max_version=_FW_V12, allow_missing=True
        )
    ).add_to_registry()
)

# v12 / v13: PowerConfiguration cluster becomes available.
(
    _apply_lixee_entities(
        QuirkBuilder(LIXEE, "ZLinky_TIC").firmware_version_filter(
            min_version=_FW_V12, max_version=_FW_V14, allow_missing=False
        )
    )
    .adds(PowerConfiguration)
    .add_to_registry()
)

# v14: add Time and the bidirectional Tuya cluster.
(
    _apply_lixee_entities(
        QuirkBuilder(LIXEE, "ZLinky_TIC").firmware_version_filter(
            min_version=_FW_V14, max_version=_FW_V15, allow_missing=False
        )
    )
    .adds(PowerConfiguration)
    .adds(Time)
    .adds(TuyaManufCluster)
    .adds(TuyaManufCluster, cluster_type=ClusterType.Client)
    .add_to_registry()
)

# v15+: same clusters as v14.
# v15 advertises the device as DIMMABLE_LIGHT, which is left as-is.
(
    _apply_lixee_entities(
        QuirkBuilder(LIXEE, "ZLinky_TIC").firmware_version_filter(
            min_version=_FW_V15, allow_missing=False
        )
    )
    .adds(PowerConfiguration)
    .adds(Time)
    .adds(TuyaManufCluster)
    .adds(TuyaManufCluster, cluster_type=ClusterType.Client)
    .add_to_registry()
)
