"""Tests for LiXee quirks."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest
from zha.quirks import DEVICE_REGISTRY
from zha.zigbee.device import DeviceEntityRemovedEvent
from zigpy.profiles import zha
import zigpy.types as t
from zigpy.zcl import ClusterType
from zigpy.zcl.clusters.general import Time
from zigpy.zcl.clusters.homeautomation import ElectricalMeasurement
from zigpy.zcl.clusters.smartenergy import Metering

import zhaquirks
from zhaquirks.builder.metadata import DeviceAlertLevel
from zhaquirks.lixee import ZLINKY_MANUFACTURER_CLUSTER_ID
from zhaquirks.lixee.zlinky import (
    ENTITIES_CHANGED_EVENT,
    SINGLE_INDEX_KEY,
    LinkyMode,
    TempoColor,
    ZLinkyTICElectricalMeasurement,
    ZLinkyTICManufacturerCluster,
    _relay_closed,
    profile_slot_key,
    tier_key,
)

zhaquirks.setup()

ZLINKY_ATTRS = ZLinkyTICManufacturerCluster.AttributeDefs
EM_ATTRS = ElectricalMeasurement.AttributeDefs


def _zlinky(zigpy_device_from_v2_quirk, firmware_version: int | None = 0x11):
    device = zigpy_device_from_v2_quirk(
        "LiXee",
        "ZLinky_TIC",
        cluster_ids={
            1: {
                Metering.cluster_id: ClusterType.Server,
                ElectricalMeasurement.cluster_id: ClusterType.Server,
                ZLINKY_MANUFACTURER_CLUSTER_ID: ClusterType.Server,
            }
        },
        firmware_version=firmware_version,
    )
    # as in the device diagnostics, for ZHA to discover its own entities
    device.endpoints[1].profile_id = zha.PROFILE_ID
    device.endpoints[1].device_type = zha.DeviceType.DIMMABLE_LIGHT
    return device


def _cancelled_task(coro, *args, **kwargs) -> asyncio.Task:
    """Do not run periodic entity refreshes."""
    task = asyncio.get_running_loop().create_task(coro)
    task.cancel()
    return task


def _zha_device(device):
    """Return the ZHA device of the quirk, with a mock gateway."""
    gateway = mock.MagicMock()
    gateway.config.config.device_options.consider_unavailable_mains = 7200
    gateway.config.config.device_options.consider_unavailable_battery = 21600
    gateway.async_create_background_task.side_effect = _cancelled_task
    return DEVICE_REGISTRY.match_entry(device).zha_device_factory(device, gateway)


def _converters(device) -> dict:
    """Return the attribute converters of the quirk entities, by unique_id suffix."""
    entry = DEVICE_REGISTRY.match_entry(device)
    return {
        metadata.resolved_unique_id_suffix: getattr(
            metadata, "attribute_converter", None
        )
        for metadata in entry.zha_device_factory.quirk_definition.entity_metadata
    }


@pytest.mark.parametrize("firmware_version", [0x11, None])
def test_zlinky_quirk_applied(zigpy_device_from_v2_quirk, firmware_version):
    """Test the quirk applies with a recent or an unknown firmware version."""
    device = _zlinky(zigpy_device_from_v2_quirk, firmware_version)
    cluster = device.endpoints[1].zlinky_manufacturer_specific
    assert isinstance(cluster, ZLinkyTICManufacturerCluster)
    assert (
        device.endpoints[1].smartenergy_metering.get(Metering.AttributeDefs.divisor.id)
        == 1000
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "E260701015506",
            datetime(2026, 7, 1, 1, 55, 6, tzinfo=timezone(timedelta(hours=2))),
        ),
        (
            "H081225223518",
            datetime(2008, 12, 25, 22, 35, 18, tzinfo=timezone(timedelta(hours=1))),
        ),
        # lowercase season: meter clock in degraded mode
        (
            b"e090714074553",
            datetime(2009, 7, 14, 7, 45, 53, tzinfo=timezone(timedelta(hours=2))),
        ),
        (" 090714074553", None),
        ("E261301015506", None),
        ("E2607", None),
        ("", None),
    ],
)
def test_zlinky_date(zigpy_device_from_v2_quirk, raw, expected):
    """Test the TIC timestamp is parsed into an aware datetime."""
    converters = _converters(_zlinky(zigpy_device_from_v2_quirk))
    assert converters[ZLINKY_ATTRS.std_current_date_and_time.name](raw) == expected


def test_zlinky_text(zigpy_device_from_v2_quirk):
    """Test the TIC strings are stripped of their padding spaces."""
    converters = _converters(_zlinky(zigpy_device_from_v2_quirk))
    assert converters[ZLINKY_ATTRS.linky_tariff_period.name]("      BASE") == "BASE"
    assert converters[ZLINKY_ATTRS.linky_tariff_period.name]("   ") is None
    message = converters[ZLINKY_ATTRS.std_message_short.name]
    assert message("COUPURE  PREVUE   LE 12/11") == "COUPURE PREVUE LE 12/11"
    # no message is shown as empty
    assert message("PAS DE          MESSAGE") == ""
    assert message("   ") == ""
    # firmware buffers: NUL padded, with stale bytes of a previous value
    assert message("\x00" * 32) == ""
    assert message("ALERTE\x00 PREVUE LE 12/11") == "ALERTE"
    assert converters[ZLINKY_ATTRS.linky_tariff_period.name](b"BASE\x00\x00XX") == (
        "BASE"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            b"003A0001",
            {
                "cut_off_switch": "closed",
                "cut_off_switch_open": False,
                "reference_power_exceeded": False,
                "overvoltage": False,
                "terminal_cover": False,
                "cpl_status": "new_lock",
                "cpl_synchronized": False,
                "injecting": False,
                "clock_degraded": False,
                "distributor_tariff_index": 1,
                "mobile_peak_notice": "none",
                "mobile_peak": "none",
                ZLINKY_ATTRS.std_status_register.name: "003A0001",
            },
        ),
        (
            # overpower cut, terminal cover open, overvoltage, power exceeded,
            # injecting, distributor index 4, clock degraded, PM1 notice,
            # PM1 in progress
            b"5E01C2D2",
            {
                "cut_off_switch": "open_overpower",
                "cut_off_switch_open": True,
                "reference_power_exceeded": True,
                "overvoltage": True,
                "terminal_cover": True,
                "cpl_status": "new_unlock",
                "cpl_synchronized": False,
                "injecting": True,
                "clock_degraded": True,
                "distributor_tariff_index": 4,
                "mobile_peak_notice": "pm1",
                "mobile_peak": "pm1",
            },
        ),
        (
            # CPL registered and synchronized
            b"00C00000",
            {"cpl_status": "registered", "cpl_synchronized": True},
        ),
        (
            b"0000000E",
            # undefined reason: still not closed
            {"cut_off_switch": None, "cut_off_switch_open": True},
        ),
        *(
            # not 8 hex digits, though `int(raw, 16)` would accept them
            (raw, {"cut_off_switch": None, "reference_power_exceeded": None})
            for raw in (b"0x3A0001", b"-03A0001", b"003A_001")
        ),
        (
            # legacy cached value, from when STGE was decoded as a string
            "b'003A0001'",
            {
                "cut_off_switch": None,
                "cut_off_switch_open": None,
                "reference_power_exceeded": None,
            },
        ),
    ],
)
def test_zlinky_status_register(zigpy_device_from_v2_quirk, raw, expected):
    """Test the STGE status register fields."""
    converters = _converters(_zlinky(zigpy_device_from_v2_quirk))
    for suffix, value in expected.items():
        assert converters[suffix](raw) == value, suffix


def test_zlinky_supplier_indexes_follow_usage(zigpy_device_from_v2_quirk):
    """Test the supplier indexes only get an entity once they report energy."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific
    metering = device.endpoints[1].smartenergy_metering

    def tier_attr(index):
        return getattr(Metering.AttributeDefs, f"current_tier{index}_summ_delivered")

    def created():
        return {
            entity._attribute_name: entity
            for entity in _zha_device(device).discover_entities()
            if entity.PLATFORM == "sensor"
            and str(getattr(entity, "_attribute_name", "")).startswith("current_tier")
        }

    with mock.patch.object(cluster, "refresh_non_reportable"):
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.standard_single_phase
        )
        # only indexes 1, 2, 7 and 9 report energy
        for index, value in (
            (1, 700),
            (2, 300),
            (3, 0),
            (4, 0),
            (5, None),
            (7, 1000),
            (8, 0),
            (9, 5000),
            (10, None),
        ):
            if value is not None:
                metering.update_attribute(tier_attr(index).id, value)
        before = cluster._irrelevant_attribute_ids()
        entities = created()
        assert set(entities) == {tier_attr(index).name for index in (1, 2, 7, 9)}
        # shown means enabled: nobody has to enable them by hand
        assert all(e.entity_registry_enabled_default for e in entities.values())

        # the meter starts using index 8: the configuration change is detected
        metering.update_attribute(tier_attr(8).id, 200)
        after = cluster._irrelevant_attribute_ids()
        assert before - after == {tier_key(8)}
        assert tier_attr(8).name in created()


def test_zlinky_relays(zigpy_device_from_v2_quirk):
    """Test the RELAIS bitmap is split per relay."""
    # relays 3, 4 and 8 closed (example from the Enedis specification)
    assert {number for number in range(1, 9) if _relay_closed(number)(140)} == {
        3,
        4,
        8,
    }
    dry_contact = _converters(_zlinky(zigpy_device_from_v2_quirk))[
        ZLINKY_ATTRS.std_relay.name
    ]
    assert dry_contact(1) is True
    assert dry_contact(140) is False


def test_zlinky_day_profile(zigpy_device_from_v2_quirk):
    """Test the next day profile is summarized as tariff index changes."""
    converters = _converters(_zlinky(zigpy_device_from_v2_quirk))
    convert = converters[ZLINKY_ATTRS.std_supplier_calendar_next_day_profile.name]
    nonutile = " NONUTILE" * 8
    assert convert("00004001" + " NONUTILE" * 10) == "00:00 index 1"
    # hour and minute are read as decimal digits
    assert (
        convert("00004001 06304002 16590000" + nonutile)
        == "00:00 index 1; 06:30 index 2"
    )
    # real heures pleines/creuses profile
    assert (
        convert("0000C001 06008002 1600C001" + " NONUTILE" * 8)
        == "00:00 index 1, dry contact closed; "
        "06:00 index 2, dry contact open; "
        "16:00 index 1, dry contact closed"
    )
    assert convert(" NONUTILE" * 11) is None
    # bits 4-10 are the virtual contacts 1-7, with or without an index change,
    # and clearing them all is a change too
    assert convert("00008051 06000032 10000040 12000001" + " NONUTILE" * 7) == (
        "00:00 index 1, dry contact open, contacts 1+3; "
        "06:00 index 2, contacts 1+2; "
        "10:00 contacts 3; "
        "12:00 index 1, contacts none"
    )
    assert convert("00000010 06000000" + " NONUTILE" * 6) == (
        "00:00 contacts 1; 06:00 contacts none"
    )
    # the documented grammar parses back to the slots
    slots = []
    for slot in convert("00008051 06000032 10000040 12000001" + " NONUTILE" * 7).split(
        "; "
    ):
        time, _, rest = slot.partition(" ")
        slots.append((time, rest.split(", ")))
    assert slots == [
        ("00:00", ["index 1", "dry contact open", "contacts 1+3"]),
        ("06:00", ["index 2", "contacts 1+2"]),
        ("10:00", ["contacts 3"]),
        ("12:00", ["index 1", "contacts none"]),
    ]
    # cut at a slot boundary to fit a Home Assistant state
    summary = convert(" ".join(f"{hour:02}00FFF1" for hour in range(24)))
    assert len(summary) <= 255
    assert summary.startswith("00:00 index 1, dry contact closed, contacts 1+2+3")
    assert summary.endswith("contacts 1+2+3+4+5+6+7")
    # bits 15-14: 10 opens, 11 closes, 00 leaves and 01 (Tempo config) is ignored
    assert convert("00008001 01004001 02000001" + " NONUTILE" * 8) == (
        "00:00 index 1, dry contact open; 01:00 index 1; 02:00 index 1"
    )
    # impossible or hex start times are ignored
    assert convert("24008001 0A008001 12608001" + " NONUTILE" * 8) is None


def test_zlinky_day_profile_slots(zigpy_device_from_v2_quirk):
    """Test each day profile slot is split into start, index and contacts."""
    converters = _converters(_zlinky(zigpy_device_from_v2_quirk))

    def slots(raw, prefix="pjourf_plus_1"):
        return [
            tuple(
                converters[f"{prefix}_slot_{slot}_{field}"](raw)
                for field in ("start", "index", "contacts")
            )
            for slot in range(1, 9)
        ]

    # real heures pleines/creuses profile: 0 is the dry contact
    assert slots("0000C001 06008002 1600C001" + " NONUTILE" * 5) == [
        ("00:00", 1, "0"),
        ("06:00", 2, ""),
        ("16:00", 1, "0"),
        *[(None, None, None)] * 5,
    ]
    # real profile with split off-peak hours, times are decimal
    assert slots(
        "00008002 0122C001 06228002 1352C001 16528002 NONUTILE NONUTILE NONUTILE"
    ) == [
        ("00:00", 2, ""),
        ("01:22", 1, "0"),
        ("06:22", 2, ""),
        ("13:52", 1, "0"),
        ("16:52", 2, ""),
        *[(None, None, None)] * 3,
    ]
    # unchanged index and dry contact are kept, virtual contacts are a state,
    # and slots changing nothing are omitted
    assert slots("0000C051 06000032 10000032 12000000 13000001", "ppointe1")[:5] == [
        ("00:00", 1, "0,1,3"),
        ("06:00", 2, "0,1,2"),
        ("12:00", 2, "0"),
        ("13:00", 1, "0"),
        (None, None, None),
    ]
    # an action changing nothing first is not a slot
    assert slots("00000000 06000002")[:2] == [("06:00", 2, ""), (None, None, None)]
    assert slots(" NONUTILE" * 8) == [(None, None, None)] * 8
    assert slots(None) == [(None, None, None)] * 8


def test_zlinky_day_profile_slots_follow_profile(zigpy_device_from_v2_quirk):
    """Test only the slots the meter sent get entities, updated with the profile."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific
    profile = ZLINKY_ATTRS.std_supplier_calendar_next_day_profile.id
    zha_device = _zha_device(device)
    # entities are recomputed by hand below
    zha_device.gateway.async_create_task.side_effect = lambda coro, **kwargs: (
        coro.close()
    )
    notify_mock = mock.Mock()
    cluster.on_event(ENTITIES_CHANGED_EVENT, notify_mock)

    def created():
        return {
            entity.unique_id.rsplit("-", 1)[-1]: entity
            for entity in zha_device.discover_entities()
            if "_slot_" in entity.unique_id
        }

    with mock.patch.object(cluster, "refresh_non_reportable"):
        # the slots are hidden until the profile is known, even in unknown mode
        assert created() == {}
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.standard_single_phase
        )
        cluster.update_attribute(profile, _BASE_PROFILE)
        notify_mock.assert_called_once()
        entities = created()
        assert set(entities) == {
            f"pjourf_plus_1_slot_1_{field}" for field in ("start", "index", "contacts")
        }
        assert not any(e.entity_registry_enabled_default for e in entities.values())

        # tomorrow has 3 slots: the entities are updated at once
        notify_mock.reset_mock()
        cluster.update_attribute(profile, _HC_PROFILE)
        notify_mock.assert_called_once()
        assert {int(suffix.split("_")[4]) for suffix in created()} == {1, 2, 3}

        # the same profile again changes nothing
        notify_mock.reset_mock()
        created()
        cluster.update_attribute(profile, _HC_PROFILE)
        notify_mock.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "historique"),
    [
        (LinkyMode.historique_single_phase, True),
        (LinkyMode.standard_single_phase, False),
        (LinkyMode.historique_three_phase, True),
        (LinkyMode.standard_three_phase_producer, False),
    ],
)
def test_zlinky_historique_mode(zigpy_device_from_v2_quirk, mode, historique):
    """Test the historique mode problem sensor."""
    converters = _converters(_zlinky(zigpy_device_from_v2_quirk))
    assert converters["historique_mode"](mode) is historique


async def test_zlinky_refresh_non_reportable(zigpy_device_from_v2_quirk):
    """Test non-reportable attributes are re-read on reports, at most once a minute."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    metering = device.endpoints[1].smartenergy_metering
    cluster = device.endpoints[1].zlinky_manufacturer_specific

    async def flush():
        while cluster._tasks:
            await asyncio.gather(*cluster._tasks)

    def read_names():
        names = [name for call in read_mock.call_args_list for name in call.args[0]]
        read_mock.reset_mock()
        return names

    with (
        mock.patch.object(cluster, "read_attributes", autospec=True) as read_mock,
        mock.patch("zhaquirks.lixee.zlinky.time") as time_mock,
    ):
        time_mock.monotonic.return_value = 1000.0

        # mode unknown: only the attributes common to both modes are read
        metering.update_attribute(
            Metering.AttributeDefs.current_summ_delivered.id, 12201584
        )
        await flush()
        assert read_names() == [
            ZLINKY_ATTRS.linky_mode.name,
            ZLINKY_ATTRS.linky_tariff_period.name,
            ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name.name,
        ]
        assert all(
            call.kwargs == {"allow_cache": False} for call in read_mock.call_args_list
        )

        # standard mode becomes known: standard attributes are read
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.standard_single_phase
        )
        await flush()
        names = read_names()
        assert ZLINKY_ATTRS.std_current_date_and_time.name in names
        assert ZLINKY_ATTRS.std_status_register.name in names
        assert ZLINKY_ATTRS.std_relay.name in names
        assert ZLINKY_ATTRS.std_supplier_calendar_next_day_profile.name in names
        assert ZLINKY_ATTRS.linky_status.name not in names
        # not relevant to a single-phase consumer meter
        assert ZLINKY_ATTRS.std_apparent_power_injected_max_1.name not in names
        assert (
            ZLINKY_ATTRS.std_apparent_power_withdrawn_max_phase_2_n_1.name not in names
        )
        assert ZLINKY_ATTRS.std_apparent_power_withdrawn_max_phase_1_n_1.name in names
        easd = ZLINKY_ATTRS.std_active_energy_withdrawn_distributor_index_01.name
        assert easd in names

        # throttled until the interval elapsed
        metering.update_attribute(
            Metering.AttributeDefs.current_summ_delivered.id, 12201585
        )
        await flush()
        assert read_names() == []

        # after a minute, only the fast groups are read again
        time_mock.monotonic.return_value = 1061.0
        metering.update_attribute(
            Metering.AttributeDefs.current_summ_delivered.id, 12201586
        )
        await flush()
        names = read_names()
        assert ZLINKY_ATTRS.std_current_date_and_time.name in names
        assert ZLINKY_ATTRS.std_supplier_calendar_next_day_profile.name not in names
        assert easd not in names

        # the distributor indexes every 15 minutes, in case they are not reported
        time_mock.monotonic.return_value = 1000.0 + 900
        metering.update_attribute(
            Metering.AttributeDefs.current_summ_delivered.id, 12201587
        )
        await flush()
        names = read_names()
        assert easd in names
        assert ZLINKY_ATTRS.std_supplier_calendar_next_day_profile.name not in names


async def test_zlinky_tempo_colors(zigpy_device_from_v2_quirk):
    """Test the Tempo colors are decoded from STGE and never read from the device."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific

    with mock.patch.object(cluster, "refresh_non_reportable"):
        # white today, red tomorrow
        cluster.update_attribute(ZLINKY_ATTRS.std_status_register.id, b"0E000000")
    assert cluster.get(ZLINKY_ATTRS.tempo_color_today.id) == TempoColor.white
    assert cluster.get(ZLINKY_ATTRS.tempo_color_tomorrow.id) == TempoColor.red

    with mock.patch.object(
        cluster, "read_attributes_raw", autospec=True
    ) as read_raw_mock:
        success, failure = await cluster.read_attributes(
            [
                ZLINKY_ATTRS.tempo_color_today.name,
                ZLINKY_ATTRS.tempo_color_tomorrow.name,
            ],
            allow_cache=False,
        )
    read_raw_mock.assert_not_called()
    assert success == {
        ZLINKY_ATTRS.tempo_color_today.name: TempoColor.white,
        ZLINKY_ATTRS.tempo_color_tomorrow.name: TempoColor.red,
    }
    assert failure == {}


@pytest.mark.parametrize(
    ("cache", "visible", "hidden"),
    [
        (
            # mode unknown: nothing is hidden
            {},
            {
                ZLINKY_ATTRS.std_relay,
                ZLINKY_ATTRS.hist_tomorrow_color,
                ZLINKY_ATTRS.hist_ejp_start_notice,
                ZLINKY_ATTRS.std_apparent_power_withdrawn_max_phase_2_n_1,
                ZLINKY_ATTRS.std_apparent_power_injected_instantaneous,
                ZLINKY_ATTRS.tempo_color_today,
                EM_ATTRS.rms_current_ph_b,
                Metering.AttributeDefs.current_summ_received,
            },
            set(),
        ),
        (
            # standard single-phase consumer, no Tempo contract (dump values)
            {
                ZLINKY_ATTRS.linky_mode: LinkyMode.standard_single_phase,
                ZLINKY_ATTRS.std_status_register: b"003A0001",
            },
            {
                ZLINKY_ATTRS.std_relay,
                ZLINKY_ATTRS.std_status_register,
                ZLINKY_ATTRS.std_apparent_power_withdrawn_max_phase_1_n_1,
                ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name,
                ZLINKY_ATTRS.linky_tariff_period,
                EM_ATTRS.rms_current,
                EM_ATTRS.active_power_ph_b,  # CCASN-1, not phase 2
                Metering.AttributeDefs.current_summ_delivered,
            },
            {
                ZLINKY_ATTRS.linky_status,
                ZLINKY_ATTRS.hist_tomorrow_color,
                ZLINKY_ATTRS.hist_subscribed_power_exceeding_warning,
                ZLINKY_ATTRS.std_apparent_power_withdrawn_max_phase_2_n_1,
                ZLINKY_ATTRS.std_apparent_power_injected_instantaneous,
                ZLINKY_ATTRS.std_injected_active_load_curve_point_n_1,
                ZLINKY_ATTRS.tempo_color_today,
                ZLINKY_ATTRS.tempo_color_tomorrow,
                EM_ATTRS.rms_current_ph_b,
                EM_ATTRS.rms_voltage_ph_c,
                Metering.AttributeDefs.current_summ_received,
            },
        ),
        (
            # standard three-phase producer, Tempo contract (blue day)
            {
                ZLINKY_ATTRS.linky_mode: LinkyMode.standard_three_phase_producer,
                ZLINKY_ATTRS.std_status_register: b"01000000",
            },
            {
                ZLINKY_ATTRS.std_apparent_power_withdrawn_max_phase_3_n_1,
                ZLINKY_ATTRS.std_apparent_power_injected_instantaneous,
                ZLINKY_ATTRS.tempo_color_today,
                ZLINKY_ATTRS.tempo_color_tomorrow,
                EM_ATTRS.rms_current_ph_b,
                Metering.AttributeDefs.current_summ_received,
            },
            {
                ZLINKY_ATTRS.hist_tomorrow_color,
                ZLINKY_ATTRS.hist_current_exceeding_warning_phase_2,
            },
        ),
        (
            # historique single-phase, Tempo contract
            {
                ZLINKY_ATTRS.linky_mode: LinkyMode.historique_single_phase,
                ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name: (
                    "BBR("
                ),
            },
            {
                ZLINKY_ATTRS.hist_tomorrow_color,
                ZLINKY_ATTRS.hist_subscribed_power_exceeding_warning,
                ZLINKY_ATTRS.linky_status,
                ZLINKY_ATTRS.linky_tariff_period,
            },
            {
                ZLINKY_ATTRS.hist_ejp_start_notice,
                ZLINKY_ATTRS.hist_current_exceeding_warning_phase_1,
                ZLINKY_ATTRS.hist_potentials_presence,
                ZLINKY_ATTRS.std_relay,
                ZLINKY_ATTRS.std_current_date_and_time,
                ZLINKY_ATTRS.tempo_color_today,
                EM_ATTRS.rms_current_ph_b,
                Metering.AttributeDefs.current_summ_received,
            },
        ),
        (
            # historique three-phase, EJP contract
            {
                ZLINKY_ATTRS.linky_mode: LinkyMode.historique_three_phase,
                ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name: (
                    "EJP."
                ),
            },
            {
                ZLINKY_ATTRS.hist_ejp_start_notice,
                ZLINKY_ATTRS.hist_current_exceeding_warning_phase_3,
                ZLINKY_ATTRS.hist_potentials_presence,
                EM_ATTRS.rms_current_ph_c,
            },
            {
                ZLINKY_ATTRS.hist_tomorrow_color,
                ZLINKY_ATTRS.std_status_register,
                # three-phase meters warn per phase (ADIR1-3)
                ZLINKY_ATTRS.hist_subscribed_power_exceeding_warning,
            },
        ),
        (
            # mode 0 before the first frame: not trusted without OPTARIF
            {ZLINKY_ATTRS.linky_mode: LinkyMode.historique_single_phase},
            {
                ZLINKY_ATTRS.std_relay,
                ZLINKY_ATTRS.std_apparent_power_withdrawn_max_phase_2_n_1,
                ZLINKY_ATTRS.tempo_color_today,
                EM_ATTRS.rms_current_ph_b,
            },
            set(),
        ),
        (
            # padding only: still no OPTARIF
            {
                ZLINKY_ATTRS.linky_mode: LinkyMode.historique_single_phase,
                ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name: (
                    "\x00" * 4
                ),
            },
            {ZLINKY_ATTRS.std_relay, EM_ATTRS.rms_current_ph_b},
            set(),
        ),
    ],
)
def test_zlinky_irrelevant_attributes_unsupported(
    zigpy_device_from_v2_quirk, cache, visible, hidden
):
    """Test attributes irrelevant to the meter configuration are hidden."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    endpoint = device.endpoints[1]
    cluster = endpoint.zlinky_manufacturer_specific
    assert isinstance(endpoint.electrical_measurement, ZLinkyTICElectricalMeasurement)

    with mock.patch.object(cluster, "refresh_non_reportable"):
        for attr, value in cache.items():
            cluster.update_attribute(attr.id, value)

    def unsupported(attr) -> bool:
        if attr in ElectricalMeasurement.attributes.values():
            return endpoint.electrical_measurement.is_attribute_unsupported(attr.name)
        if attr in Metering.attributes.values():
            return endpoint.smartenergy_metering.is_attribute_unsupported(attr.name)
        return cluster.is_attribute_unsupported(attr.name)

    assert {attr.name for attr in visible if unsupported(attr)} == set()
    assert {attr.name for attr in hidden if not unsupported(attr)} == set()


async def test_zlinky_hourly_configuration_check(zigpy_device_from_v2_quirk):
    """Test entities are recomputed hourly if the meter configuration changed."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific

    async def report(monotonic: float) -> None:
        time_mock.monotonic.return_value = monotonic
        cluster.update_attribute(
            ZLINKY_ATTRS.std_current_date_and_time.id, "E260701015506"
        )
        while cluster._tasks:
            await asyncio.gather(*cluster._tasks)

    notify_mock = mock.Mock()
    cluster.on_event(ENTITIES_CHANGED_EVENT, notify_mock)

    with (
        mock.patch.object(cluster, "read_attributes", autospec=True),
        mock.patch("zhaquirks.lixee.zlinky.time") as time_mock,
    ):
        time_mock.monotonic.return_value = 0.0
        cluster.update_attribute(
            ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name.id,
            "HC..",
        )
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.historique_single_phase
        )
        # ZHA creates the entities for a historique meter
        assert cluster.is_attribute_unsupported(ZLINKY_ATTRS.std_relay.name)

        # unchanged configuration after an hour: nothing to do
        await report(3600.0)
        notify_mock.assert_not_called()

        # the meter starts reporting energy on index 3: acted upon at the next
        # hourly check only
        device.endpoints[1].smartenergy_metering.update_attribute(
            Metering.AttributeDefs.current_tier3_summ_delivered.id, 10
        )
        await report(3600.0 + 1800)
        notify_mock.assert_not_called()

        await report(2 * 3600.0)
        notify_mock.assert_called_once()

        # ZHA re-evaluating the entities records the new configuration
        assert cluster.is_attribute_unsupported(ZLINKY_ATTRS.std_relay.name)
        await report(3 * 3600.0)
        notify_mock.assert_called_once()


async def test_zlinky_mode_change_checks_configuration(zigpy_device_from_v2_quirk):
    """Test a TIC mode change re-reads every attribute, then checks at once."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific
    option = ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name
    notify_mock = mock.Mock()
    cluster.on_event(ENTITIES_CHANGED_EVENT, notify_mock)
    checked_option = []

    async def read(attributes, **kwargs):
        # OPTARIF is replaced by NGTF, sharing attribute 0x0000
        if option.name in attributes:
            cluster.update_attribute(option.id, "H PLEINE/CREUSE")
        return {}, {}

    def check():
        checked_option.append(cluster.get(option.id))
        original_check()

    original_check = cluster._check_meter_configuration
    with (
        mock.patch.object(cluster, "read_attributes", side_effect=read) as read_mock,
        mock.patch.object(cluster, "_check_meter_configuration", side_effect=check),
        mock.patch("zhaquirks.lixee.zlinky.time") as time_mock,
    ):
        time_mock.monotonic.return_value = 100.0
        cluster.update_attribute(option.id, "HC..")
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.historique_single_phase
        )
        while cluster._tasks:
            await asyncio.gather(*cluster._tasks)
        assert cluster.is_attribute_unsupported(ZLINKY_ATTRS.std_relay.name)
        read_mock.reset_mock()

        # within the hour: the standard attributes and 0x0000 are read at once
        time_mock.monotonic.return_value = 200.0
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.standard_single_phase
        )
        while cluster._tasks:
            await asyncio.gather(*cluster._tasks)

    names = {name for call in read_mock.call_args_list for name in call.args[0]}
    assert option.name in names
    assert ZLINKY_ATTRS.std_supplier_calendar_next_day_profile.name in names
    # checked once, after NGTF was read
    assert checked_option == ["H PLEINE/CREUSE"]
    notify_mock.assert_called_once()


async def test_zlinky_failed_refresh_retried(zigpy_device_from_v2_quirk):
    """Test a failed hourly refresh is retried before the hour elapsed."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific
    profile = ZLINKY_ATTRS.std_supplier_calendar_next_day_profile.name

    async def report(monotonic: float) -> set:
        time_mock.monotonic.return_value = monotonic
        read_mock.reset_mock()
        cluster.update_attribute(
            ZLINKY_ATTRS.std_current_date_and_time.id, "E260701015506"
        )
        while cluster._tasks:
            await asyncio.gather(*cluster._tasks)
        return {name for call in read_mock.call_args_list for name in call.args[0]}

    with (
        mock.patch.object(cluster, "read_attributes", autospec=True) as read_mock,
        mock.patch("zhaquirks.lixee.zlinky.time") as time_mock,
    ):
        time_mock.monotonic.return_value = 0.0
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.standard_single_phase
        )
        while cluster._tasks:
            await asyncio.gather(*cluster._tasks)
        # the next hourly refresh fails
        read_mock.side_effect = asyncio.TimeoutError
        assert profile in await report(3600.0)
        read_mock.side_effect = None
        # retried 10 minutes later, not an hour
        assert profile not in await report(3600.0 + 599)
        assert profile in await report(3600.0 + 600)
        # then hourly again
        assert profile not in await report(3600.0 + 1800)


async def test_zlinky_entities_changed_recomputes_zha_entities(
    zigpy_device_from_v2_quirk,
):
    """Test a configuration change makes ZHA recompute the entities."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific

    zha_device = _zha_device(device)
    tasks = []
    zha_device.gateway.async_create_task.side_effect = lambda coro, **kwargs: (
        tasks.append(asyncio.get_running_loop().create_task(coro))
    )

    with mock.patch.object(
        zha_device, "recompute_entities", autospec=True
    ) as recompute_mock:
        cluster.emit(ENTITIES_CHANGED_EVENT)
        await asyncio.gather(*tasks)
        recompute_mock.assert_awaited_once()

        # no longer listening once the ZHA device is torn down
        await zha_device.async_teardown(emit_entity_events=False)
        cluster.emit(ENTITIES_CHANGED_EVENT)
        await asyncio.gather(*tasks)
        recompute_mock.assert_awaited_once()


@pytest.mark.parametrize(
    ("cache", "created", "not_created"),
    [
        (
            # standard single-phase consumer, no Tempo contract (dump values)
            {
                ZLINKY_ATTRS.linky_mode: LinkyMode.standard_single_phase,
                ZLINKY_ATTRS.std_status_register: b"003A0001",
            },
            {
                ZLINKY_ATTRS.std_relay.name,
                ZLINKY_ATTRS.std_status_register.name,
                ZLINKY_ATTRS.std_apparent_power_withdrawn_max_phase_1_n_1.name,
                ZLINKY_ATTRS.linky_mode.name,
                EM_ATTRS.rms_current.name,
                EM_ATTRS.active_power_ph_b.name,  # CCASN-1, not phase 2
            },
            {
                ZLINKY_ATTRS.std_apparent_power_injected_max.name,
                ZLINKY_ATTRS.std_apparent_power_injected_instantaneous.name,
                ZLINKY_ATTRS.std_apparent_power_withdrawn_max_phase_2_n_1.name,
                ZLINKY_ATTRS.tempo_color_today.name,
                ZLINKY_ATTRS.hist_subscribed_power_exceeding_warning.name,
                EM_ATTRS.rms_current_ph_b.name,
                Metering.AttributeDefs.current_summ_received.name,
            },
        ),
        (
            # historique single-phase: standard-only binary sensors are hidden too
            {
                ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name: (
                    "BASE"
                ),
                ZLINKY_ATTRS.linky_mode: LinkyMode.historique_single_phase,
            },
            {
                ZLINKY_ATTRS.linky_mode.name,
                ZLINKY_ATTRS.hist_subscribed_power_exceeding_warning.name,
            },
            {
                ZLINKY_ATTRS.std_relay.name,
                ZLINKY_ATTRS.std_status_register.name,
                ZLINKY_ATTRS.std_current_date_and_time.name,
            },
        ),
    ],
)
def test_zlinky_zha_entities_follow_meter(
    zigpy_device_from_v2_quirk, cache, created, not_created
):
    """Test ZHA only creates the entities relevant to the meter."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific
    with mock.patch.object(cluster, "refresh_non_reportable"):
        for attr, value in cache.items():
            cluster.update_attribute(attr.id, value)
        # ZHA only creates its own measurement sensors with a cached value (dump)
        for attr, value in (
            (EM_ATTRS.rms_current, 3),
            (EM_ATTRS.rms_current_ph_b, 0),
            (EM_ATTRS.active_power_ph_b, 492),
        ):
            device.endpoints[1].electrical_measurement.update_attribute(attr.id, value)
        device.endpoints[1].smartenergy_metering.update_attribute(
            Metering.AttributeDefs.current_summ_received.id, 0
        )

    # same check as ZHA when adding or recomputing entities
    supported = {
        getattr(entity, "_attribute_name", None)
        for entity in _zha_device(device).discover_entities()
        if entity.is_supported()
    }
    assert created - supported == set()
    assert not_created & supported == set()


@pytest.mark.parametrize(
    ("mode", "hidden"),
    [
        (LinkyMode.standard_single_phase, True),
        (LinkyMode.standard_three_phase, True),
        (LinkyMode.standard_single_phase_producer, False),
        (LinkyMode.standard_three_phase_producer, False),
    ],
)
def test_zlinky_injecting_binary_sensor_only_for_producers(
    zigpy_device_from_v2_quirk, mode, hidden
):
    """Test the injection flag is only exposed by producer meters."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific
    with mock.patch.object(cluster, "refresh_non_reportable"):
        cluster.update_attribute(ZLINKY_ATTRS.linky_mode.id, mode)
        cluster.update_attribute(
            ZLINKY_ATTRS.std_status_register.id, t.LVBytes(b"003A0010")
        )

    suffixes = {
        entity.unique_id.rsplit("-", 1)[-1]
        for entity in _zha_device(device).discover_entities()
        if entity.PLATFORM == "binary_sensor"
    }
    assert ("injecting" not in suffixes) is hidden
    assert "clock_degraded" in suffixes


_BASE_PROFILE = "00008001" + " NONUTILE" * 7
_HC_PROFILE = "0000C001 06008002 1600C001" + " NONUTILE" * 5


@pytest.mark.parametrize(
    ("cache", "hidden"),
    [
        # standard mode: the day profile and current index tell (dump values),
        # NGTF is free text chosen by the supplier
        (
            {
                ZLINKY_ATTRS.linky_mode: LinkyMode.standard_single_phase,
                ZLINKY_ATTRS.std_supplier_calendar_next_day_profile: _BASE_PROFILE,
                ZLINKY_ATTRS.std_current_tariff_index_number: 1,
            },
            True,
        ),
        (
            {
                ZLINKY_ATTRS.linky_mode: LinkyMode.standard_single_phase,
                ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name: (
                    "BASE"
                ),
                ZLINKY_ATTRS.std_supplier_calendar_next_day_profile: _HC_PROFILE,
                ZLINKY_ATTRS.std_current_tariff_index_number: 1,
            },
            False,
        ),
        (
            {
                ZLINKY_ATTRS.linky_mode: LinkyMode.standard_single_phase,
                ZLINKY_ATTRS.std_supplier_calendar_next_day_profile: _BASE_PROFILE,
                ZLINKY_ATTRS.std_current_tariff_index_number: 2,
            },
            False,
        ),
        # profile not known yet
        (
            {
                ZLINKY_ATTRS.linky_mode: LinkyMode.standard_single_phase,
                ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name: (
                    "BASE"
                ),
            },
            False,
        ),
    ],
)
def test_zlinky_virtual_contacts_only_for_multiple_indexes(
    zigpy_device_from_v2_quirk, cache, hidden
):
    """Test the virtual contacts are only exposed with several tariff indexes."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific
    with mock.patch.object(cluster, "refresh_non_reportable"):
        for attr, value in cache.items():
            cluster.update_attribute(attr.id, value)

    suffixes = {
        entity.unique_id.rsplit("-", 1)[-1]
        for entity in _zha_device(device).discover_entities()
        if entity.PLATFORM == "binary_sensor"
    }
    virtual = {f"virtual_contact_{number}" for number in range(1, 8)}
    assert (virtual & suffixes == set()) is hidden
    assert virtual <= suffixes or hidden
    # the physical dry contact is always exposed
    assert ZLINKY_ATTRS.std_relay.name in suffixes


def test_zlinky_virtual_contacts_follow_relay_bits(zigpy_device_from_v2_quirk):
    """Test virtual contact N is relay N + 1 of the RELAIS bitmap."""
    converters = _converters(_zlinky(zigpy_device_from_v2_quirk))
    # relays 3, 4 and 8 closed: virtual contacts 2, 3 and 7
    closed = {number for number in range(1, 8) if _relay_closed(number + 1)(140)}
    assert closed == {2, 3, 7}
    assert converters[ZLINKY_ATTRS.std_relay.name](140) is False


def test_zlinky_contract_change_detected(zigpy_device_from_v2_quirk):
    """Test switching to or from a single-index contract changes the config."""
    cluster = (
        _zlinky(zigpy_device_from_v2_quirk).endpoints[1].zlinky_manufacturer_specific
    )
    profile = ZLINKY_ATTRS.std_supplier_calendar_next_day_profile.id
    option = ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name.id
    with mock.patch.object(cluster, "refresh_non_reportable"):
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.standard_single_phase
        )
        cluster.update_attribute(profile, _BASE_PROFILE)
        single = cluster._irrelevant_attribute_ids()
        cluster.update_attribute(profile, _HC_PROFILE)
        multiple = cluster._irrelevant_attribute_ids()
    # HC has 2 more slots
    assert single - multiple == {
        SINGLE_INDEX_KEY,
        profile_slot_key(profile, 2),
        profile_slot_key(profile, 3),
    }

    # historique mode: OPTARIF is an enumerated value
    with mock.patch.object(cluster, "refresh_non_reportable"):
        cluster.update_attribute(option, "BASE")
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.historique_single_phase
        )
        assert cluster.is_single_index_contract()
        cluster.update_attribute(option, "HC..")
        assert not cluster.is_single_index_contract()


async def test_zlinky_entities_follow_configuration_change(zigpy_device_from_v2_quirk):
    """Test entities are deleted and created when the meter configuration changes."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific
    with mock.patch.object(cluster, "refresh_non_reportable"):
        cluster.update_attribute(
            ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name.id,
            "BASE",
        )
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.historique_single_phase
        )

    zha_device = _zha_device(device)
    zha_device._discover_new_entities()
    await zha_device._add_pending_entities(emit_event=False)

    def attributes() -> set:
        return {
            getattr(entity, "_attribute_name", None)
            for entity in zha_device.platform_entities.values()
        }

    assert ZLINKY_ATTRS.hist_subscribed_power_exceeding_warning.name in attributes()
    assert ZLINKY_ATTRS.std_relay.name not in attributes()

    events = []
    zha_device.on_all_events(events.append)
    with mock.patch.object(cluster, "refresh_non_reportable"):
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.standard_single_phase
        )
    await zha_device.recompute_entities()

    assert ZLINKY_ATTRS.hist_subscribed_power_exceeding_warning.name not in attributes()
    assert ZLINKY_ATTRS.std_relay.name in attributes()
    # entities of the historique mode are unloaded, their registry entries kept
    unloaded = {
        event.unique_id
        for event in events
        if isinstance(event, DeviceEntityRemovedEvent) and not event.remove
    }
    assert f"{device.ieee}-1-hist_subscribed_power_exceeding_warning" in unloaded
    assert not any(
        isinstance(event, DeviceEntityRemovedEvent) and event.remove for event in events
    )


async def test_zlinky_retired_entities_deleted_at_startup(zigpy_device_from_v2_quirk):
    """Test only the registry entries of retired entities are deleted once listened to."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific
    with mock.patch.object(cluster, "refresh_non_reportable"):
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.standard_single_phase
        )

    zha_device = _zha_device(device)
    events = []
    zha_device.on_all_events(events.append)

    def removed() -> set:
        return {
            event.unique_id
            for event in events
            if isinstance(event, DeviceEntityRemovedEvent) and event.remove
        }

    with mock.patch("zha.zigbee.device.initialize_cluster_configs"):
        # devices are loaded before Home Assistant listens to them
        await zha_device.async_initialize(from_cache=True)
        assert removed() == set()

        await zha_device.async_initialize(from_cache=True)
        # entities of previous versions of the quirk
        assert removed() == {
            *(f"{device.ieee}-1-std_relay_{number}" for number in range(2, 9)),
            f"{device.ieee}-1-std_message_ultra_short",
        }

        # only once
        events.clear()
        await zha_device.async_initialize(from_cache=False)
        assert removed() == set()


async def test_zlinky_load_curve_point_renamed(zigpy_device_from_v2_quirk):
    """Test the ZHA phase 2 active power entity is named after what it reports."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    device.endpoints[1].electrical_measurement.update_attribute(
        EM_ATTRS.active_power_ph_b.id, 476
    )

    zha_device = _zha_device(device)
    zha_device._discover_new_entities()
    await zha_device._add_pending_entities(emit_event=False)

    (entity,) = (
        entity
        for entity in zha_device.platform_entities.values()
        if getattr(entity, "_attribute_name", None) == EM_ATTRS.active_power_ph_b.name
    )
    assert entity.fallback_name == "Withdrawn active load curve point (n-1)"
    assert entity.translation_key == "ccasn_1"
    # same entity as before, to keep its history
    assert entity.unique_id == f"{device.ieee}-1-2820-active_power_ph_b"


@pytest.mark.parametrize(
    ("firmware_version", "time_added"),
    [(0x0B, False), (0x0E, True), (0x11, True), (None, False)],
)
def test_zlinky_unknown_firmware_adds_no_cluster(
    zigpy_device_from_v2_quirk, firmware_version, time_added
):
    """Test clusters are only added once the firmware is known to provide them."""
    device = _zlinky(zigpy_device_from_v2_quirk, firmware_version)
    assert (Time.cluster_id in device.endpoints[1].in_clusters) is time_added


def test_zlinky_unknown_attribute_report(zigpy_device_from_v2_quirk):
    """Test a report of an attribute the quirk does not define is handled."""
    cluster = (
        _zlinky(zigpy_device_from_v2_quirk).endpoints[1].zlinky_manufacturer_specific
    )
    events = []
    cluster.on_all_events(events.append)
    # e.g. 0x0400, defined by the firmware only
    with mock.patch.object(cluster, "refresh_non_reportable") as refresh_mock:
        cluster.update_attribute(0x0400, 1234)
    refresh_mock.assert_called_once()
    assert [(event.attribute_id, event.value) for event in events] == [(0x0400, 1234)]


def test_zlinky_schedule_letter(zigpy_device_from_v2_quirk):
    """Test HHPHC is shown as its schedule letter."""
    convert = _converters(_zlinky(zigpy_device_from_v2_quirk))[
        ZLINKY_ATTRS.hist_schedule_peak_hours_off_peak_hours.name
    ]
    assert convert(ord("A")) == "A"
    # never stored by the firmware
    assert convert(0) is None


def test_zlinky_latched_warnings_disabled(zigpy_device_from_v2_quirk):
    """Test the warnings the firmware never clears are disabled by default."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    entry = DEVICE_REGISTRY.match_entry(device)
    latched = {
        ZLINKY_ATTRS.hist_ejp_start_notice.name,
        ZLINKY_ATTRS.hist_subscribed_power_exceeding_warning.name,
        ZLINKY_ATTRS.hist_current_exceeding_warning_phase_1.name,
        ZLINKY_ATTRS.hist_current_exceeding_warning_phase_2.name,
        ZLINKY_ATTRS.hist_current_exceeding_warning_phase_3.name,
    }
    metadata = {
        metadata.resolved_unique_id_suffix: metadata
        for metadata in entry.zha_device_factory.quirk_definition.entity_metadata
    }
    assert all(metadata[name].initially_disabled for name in latched)


async def test_zlinky_failed_entity_update_retried(zigpy_device_from_v2_quirk):
    """Test a failed entity update is retried at a configuration check soon."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific
    zha_device = _zha_device(device)
    notify_mock = mock.Mock()
    cluster.on_event(ENTITIES_CHANGED_EVENT, notify_mock)

    with (
        mock.patch.object(cluster, "read_attributes", autospec=True),
        mock.patch("zhaquirks.lixee.zlinky.time") as time_mock,
    ):
        time_mock.monotonic.return_value = 0.0
        cluster.update_attribute(
            ZLINKY_ATTRS.linky_mode.id, LinkyMode.standard_single_phase
        )
        while cluster._tasks:
            await asyncio.gather(*cluster._tasks)

        with mock.patch.object(
            zha_device, "recompute_entities", side_effect=RuntimeError
        ):
            await zha_device._update_entities()

        async def report(monotonic: float) -> None:
            time_mock.monotonic.return_value = monotonic
            cluster.update_attribute(
                ZLINKY_ATTRS.std_current_date_and_time.id, "E260701015506"
            )
            while cluster._tasks:
                await asyncio.gather(*cluster._tasks)

        # same configuration, but the entities are updated again
        await report(599.0)
        notify_mock.assert_not_called()
        await report(600.0)
        notify_mock.assert_called_once()


@pytest.mark.parametrize(
    ("cache", "alerted"),
    [
        (
            {
                ZLINKY_ATTRS.hist_tariff_option_or_std_supplier_price_schedule_name: (
                    "BASE"
                ),
                ZLINKY_ATTRS.linky_mode: LinkyMode.historique_three_phase,
            },
            True,
        ),
        ({ZLINKY_ATTRS.linky_mode: LinkyMode.standard_single_phase}, False),
        # mode 0 before the first frame
        ({ZLINKY_ATTRS.linky_mode: LinkyMode.historique_single_phase}, False),
        # mode unknown
        ({}, False),
    ],
)
def test_zlinky_historique_mode_alert(zigpy_device_from_v2_quirk, cache, alerted):
    """Test a warning asks to switch the TIC to standard mode."""
    device = _zlinky(zigpy_device_from_v2_quirk)
    cluster = device.endpoints[1].zlinky_manufacturer_specific
    zha_device = _zha_device(device)
    assert tuple(zha_device.device_alerts) == ()

    with mock.patch.object(cluster, "refresh_non_reportable"):
        for attr, value in cache.items():
            cluster.update_attribute(attr.id, value)

    # follows the mode, not cached
    alerts = tuple(zha_device.device_alerts)
    assert bool(alerts) is alerted
    if alerted:
        (alert,) = alerts
        assert alert.level == DeviceAlertLevel.WARNING
        assert "standard mode" in alert.message
