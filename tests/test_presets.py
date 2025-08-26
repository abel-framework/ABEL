# -*- coding: utf-8 -*-
"""
ABEL : Tests of collider presets
=======================================

This file is a part of ABEL.
Copyright 2022– C.A.Lindstrøm, J.B.B.Chen, O.G.Finnerud,
D.Kallvik, E.Hørlyk, K.N.Sjobak, E.Adli, University of Oslo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import pytest
from abel.classes.collider.preset.c3 import C3
from abel.classes.collider.preset.halhf_v1 import HALHFv1
from abel.classes.collider.preset.halhf_v2 import HALHFv2
import shutil
import numpy as np
from matplotlib import pyplot as plt


@pytest.mark.presets
def test_C3():
    """
    Test for the C3 collider preset.
    """

    np.random.seed(42)

    cool_copper_collider = C3()
    cool_copper_collider.run('test_C3', overwrite=True, verbose=False)

    # Tests
    assert np.isclose(cool_copper_collider.bunch_charge, 1e-9, rtol=1e-15, atol=0.0)
    assert isinstance(cool_copper_collider.bunch_separation, float) and cool_copper_collider.bunch_separation > 0.0
    assert isinstance(cool_copper_collider.bunch_separation_ns, float) and cool_copper_collider.bunch_separation_ns > 0.0
    assert isinstance(cool_copper_collider.carbon_tax_cost(), float) and cool_copper_collider.carbon_tax_cost() > 0.0
    assert isinstance(cool_copper_collider.collision_rate(), float) and cool_copper_collider.collision_rate() > 0.0
    assert np.isclose(cool_copper_collider.com_energy, 250e9, rtol=1e-15, atol=0.0)
    assert isinstance(cool_copper_collider.construction_cost(), float) and cool_copper_collider.construction_cost() > 0.0
    assert isinstance(cool_copper_collider.construction_emissions(), float) and cool_copper_collider.construction_emissions() > 0.0
    assert isinstance(cool_copper_collider.cost_carbon_tax_per_emissions, float) and cool_copper_collider.cost_carbon_tax_per_emissions > 0.0
    assert isinstance(cool_copper_collider.cost_eu_accounting(), float) and cool_copper_collider.cost_eu_accounting() > 0.0
    assert isinstance(cool_copper_collider.cost_factor_infrastructure_and_services, float) and cool_copper_collider.cost_factor_infrastructure_and_services > 0.0
    assert isinstance(cool_copper_collider.cost_per_energy, float) and cool_copper_collider.cost_per_energy > 0.0
    assert isinstance(cool_copper_collider.cost_per_ip, float) and cool_copper_collider.cost_per_ip > 0.0
    assert isinstance(cool_copper_collider.cost_per_length_bds, float) and cool_copper_collider.cost_per_length_bds > 0.0
    assert isinstance(cool_copper_collider.cost_per_length_combiner_ring, float) and cool_copper_collider.cost_per_length_combiner_ring > 0.0
    assert isinstance(cool_copper_collider.cost_per_length_damping_ring, float) and cool_copper_collider.cost_per_length_damping_ring > 0.0
    assert isinstance(cool_copper_collider.cost_per_length_interstage, float) and cool_copper_collider.cost_per_length_interstage > 0.0
    assert isinstance(cool_copper_collider.cost_per_length_plasma_stage, float) and cool_copper_collider.cost_per_length_plasma_stage > 0.0
    assert isinstance(cool_copper_collider.cost_per_length_rf_structure_normalconducting, float) and cool_copper_collider.cost_per_length_rf_structure_normalconducting > 0.0
    assert isinstance(cool_copper_collider.cost_per_length_rf_structure_superconducting, float) and cool_copper_collider.cost_per_length_rf_structure_superconducting > 0.0
    assert isinstance(cool_copper_collider.cost_per_length_surfacebuilding, float) and cool_copper_collider.cost_per_length_surfacebuilding > 0.0
    assert isinstance(cool_copper_collider.cost_per_length_transfer_line, float) and cool_copper_collider.cost_per_length_transfer_line > 0.0
    assert isinstance(cool_copper_collider.cost_per_length_tunnel_large, float) and cool_copper_collider.cost_per_length_tunnel_large > 0.0
    assert isinstance(cool_copper_collider.cost_per_source, float) and cool_copper_collider.cost_per_source > 0.0
    assert isinstance(cool_copper_collider.cost_per_source_polarized_electrons, float) and cool_copper_collider.cost_per_source_polarized_electrons > 0.0
    assert isinstance(cool_copper_collider.cost_per_source_polarized_positrons, float) and cool_copper_collider.cost_per_source_polarized_positrons > 0.0
    assert isinstance(cool_copper_collider.cost_snowmass_itf_accounting(), float) and cool_copper_collider.cost_snowmass_itf_accounting() > 0.0
    assert isinstance(cool_copper_collider.emissions_per_energy_usage, float) and cool_copper_collider.emissions_per_energy_usage > 0.0
    assert np.isclose(cool_copper_collider.energy_asymmetry, 1.0, rtol=1e-15, atol=0.0)
    assert isinstance(cool_copper_collider.energy_cost(), float) and cool_copper_collider.energy_cost() > 0.0
    assert isinstance(cool_copper_collider.energy_emissions(), float) and cool_copper_collider.energy_emissions() > 0.0
    assert isinstance(cool_copper_collider.energy_usage(), float) and cool_copper_collider.energy_usage() > 0.0
    assert np.isclose(cool_copper_collider.enhancement_factor(), 1.0, rtol=1e-15, atol=0.0)
    assert isinstance(cool_copper_collider.full_luminosity(), float) and cool_copper_collider.full_luminosity() > 1e37
    assert isinstance(cool_copper_collider.full_luminosity_per_crossing(), float) and cool_copper_collider.full_luminosity_per_crossing() > 1e33
    assert isinstance(cool_copper_collider.full_luminosity_per_power(), float) and cool_copper_collider.full_luminosity_per_power() > 1e29
    assert isinstance(cool_copper_collider.full_programme_cost(), float) and cool_copper_collider.full_programme_cost() > 0.0
    assert isinstance(cool_copper_collider.geometric_luminosity_per_crossing(), float) and cool_copper_collider.geometric_luminosity_per_crossing() > 0.0
    assert np.isclose(cool_copper_collider.get_com_energy(), 250e9, rtol=1e-15, atol=0.0)
    assert isinstance(cool_copper_collider.get_cost(), float) and cool_copper_collider.get_cost() > 0.0
    #cool_copper_collider.get_cost_breakdown()
    #cool_copper_collider.get_cost_breakdown_overheads()
    #cool_copper_collider.get_cost_civil_construction()
    #event = cool_copper_collider.get_event()
    assert np.isclose(cool_copper_collider.get_energy_asymmetry(), 1.0, rtol=1e-15, atol=0.0)
    assert isinstance(cool_copper_collider.integrated_energy_usage(), float) and cool_copper_collider.integrated_energy_usage() > 0.0
    assert isinstance(cool_copper_collider.integrated_runtime(), float) and cool_copper_collider.integrated_runtime() > 0.0
    assert cool_copper_collider.is_scan() is False
    assert isinstance(cool_copper_collider.learning_curve_klystrons, float) and cool_copper_collider.learning_curve_klystrons >= 0.0
    assert isinstance(cool_copper_collider.length_end_to_end(), float) and cool_copper_collider.length_end_to_end() > 0.0
    assert isinstance(cool_copper_collider.linac_gradient, float) and cool_copper_collider.linac_gradient > 0.0
    assert isinstance(cool_copper_collider.maintenance_cost(), float) and cool_copper_collider.maintenance_cost() > 0.0
    assert isinstance(cool_copper_collider.maintenance_labor(), float) and cool_copper_collider.maintenance_labor() > 0.0
    assert isinstance(cool_copper_collider.maintenance_labor_per_construction_cost, float) and cool_copper_collider.maintenance_labor_per_construction_cost > 0.0
    #cool_copper_collider.maximum_upsilon()  # TODO: why does this give FALSE?
    assert isinstance(cool_copper_collider.num_bunches_in_train, int) and cool_copper_collider.num_bunches_in_train > 0
    #cool_copper_collider.num_coherent_pairs()  # TODO: why does this give FALSE?
    #cool_copper_collider.num_photons_beam1()  # TODO: why does this give FALSE?
    #cool_copper_collider.num_photons_beam2()  # TODO: why does this give FALSE?
    assert cool_copper_collider.num_shots == 1
    assert cool_copper_collider.num_shots_per_step == 1
    assert cool_copper_collider.num_steps == 1
    assert cool_copper_collider.num_structures_per_klystron == 130
    assert isinstance(cool_copper_collider.overhead_cost(), float) and cool_copper_collider.overhead_cost() > 0.0
    assert isinstance(cool_copper_collider.overhead_cost_design_and_development(), float) and cool_copper_collider.overhead_cost_design_and_development() > 0.0
    assert isinstance(cool_copper_collider.overhead_cost_management_inspection(), float) and cool_copper_collider.overhead_cost_management_inspection() > 0.0
    assert isinstance(cool_copper_collider.peak_luminosity(), float) and cool_copper_collider.peak_luminosity() > 0.0
    assert isinstance(cool_copper_collider.peak_luminosity_per_crossing(), float) and cool_copper_collider.peak_luminosity_per_crossing() > 0.0
    assert isinstance(cool_copper_collider.peak_luminosity_per_power(), float) and cool_copper_collider.peak_luminosity_per_power() > 0.0
    assert isinstance(cool_copper_collider.power_overhead(), float) and cool_copper_collider.power_overhead() > 0.0
    assert isinstance(cool_copper_collider.programme_duration(), float) and cool_copper_collider.programme_duration() > 0.0
    assert isinstance(cool_copper_collider.rep_rate_trains, float) and cool_copper_collider.rep_rate_trains > 0.0
    assert isinstance(cool_copper_collider.rf_frequency, float) and cool_copper_collider.rf_frequency > 0.0
    cool_copper_collider.run_name == 'test_C3'
    assert isinstance(cool_copper_collider.target_integrated_luminosity, float) and cool_copper_collider.target_integrated_luminosity > 0.0
    assert isinstance(cool_copper_collider.total_emissions(), float) and cool_copper_collider.total_emissions() > 0.0
    assert isinstance(cool_copper_collider.total_tunnel_length(), float) and cool_copper_collider.total_tunnel_length() > 0.0
    assert isinstance(cool_copper_collider.uptime_percentage, float) and cool_copper_collider.uptime_percentage > 0.0 and cool_copper_collider.uptime_percentage < 1.0
    assert isinstance(cool_copper_collider.wallplug_power(), float) and cool_copper_collider.wallplug_power() > 0.0

    #cool_copper_collider.final_beam

    # Test plotting and printing
    plt.ion()
    cool_copper_collider.plot_survey()
    
    cool_copper_collider.print_cost()
    cool_copper_collider.print_emissions()
    cool_copper_collider.print_power()

    # Remove output directory
    shutil.rmtree(cool_copper_collider.run_path())


@pytest.mark.presets
def test_HALHFv1():
    """
    Test for the HALHFv1 collider preset.
    """

    np.random.seed(42)

    halhf1 = HALHFv1()
    halhf1.run('test_HALHFv1', overwrite=True, verbose=False)

    # Tests
    assert isinstance(halhf1.bunch_separation, float) and halhf1.bunch_separation > 0.0
    assert isinstance(halhf1.carbon_tax_cost(), float) and halhf1.carbon_tax_cost() > 0.0
    assert isinstance(halhf1.collision_rate(), float) and halhf1.collision_rate() > 0.0
    assert np.isclose(halhf1.com_energy, 250e9, rtol=1e-15, atol=0.0)
    assert isinstance(halhf1.construction_cost(), float) and halhf1.construction_cost() > 0.0
    assert isinstance(halhf1.construction_emissions(), float) and halhf1.construction_emissions() > 0.0
    assert isinstance(halhf1.cost_carbon_tax_per_emissions, float) and halhf1.cost_carbon_tax_per_emissions > 0.0
    assert isinstance(halhf1.cost_eu_accounting(), float) and halhf1.cost_eu_accounting() > 0.0
    assert isinstance(halhf1.cost_factor_infrastructure_and_services, float) and halhf1.cost_factor_infrastructure_and_services > 0.0
    assert isinstance(halhf1.cost_per_energy, float) and halhf1.cost_per_energy > 0.0
    assert isinstance(halhf1.cost_per_ip, float) and halhf1.cost_per_ip > 0.0
    assert isinstance(halhf1.cost_per_length_bds, float) and halhf1.cost_per_length_bds > 0.0
    assert isinstance(halhf1.cost_per_length_combiner_ring, float) and halhf1.cost_per_length_combiner_ring > 0.0
    assert isinstance(halhf1.cost_per_length_damping_ring, float) and halhf1.cost_per_length_damping_ring > 0.0
    assert isinstance(halhf1.cost_per_length_interstage, float) and halhf1.cost_per_length_interstage > 0.0
    assert isinstance(halhf1.cost_per_length_plasma_stage, float) and halhf1.cost_per_length_plasma_stage > 0.0
    assert isinstance(halhf1.cost_per_length_rf_structure_normalconducting, float) and halhf1.cost_per_length_rf_structure_normalconducting > 0.0
    assert isinstance(halhf1.cost_per_length_rf_structure_superconducting, float) and halhf1.cost_per_length_rf_structure_superconducting > 0.0
    assert isinstance(halhf1.cost_per_length_surfacebuilding, float) and halhf1.cost_per_length_surfacebuilding > 0.0
    assert isinstance(halhf1.cost_per_length_transfer_line, float) and halhf1.cost_per_length_transfer_line > 0.0
    assert isinstance(halhf1.cost_per_length_tunnel_large, float) and halhf1.cost_per_length_tunnel_large > 0.0
    assert isinstance(halhf1.cost_per_length_turnaround, float) and halhf1.cost_per_length_turnaround > 0.0
    assert isinstance(halhf1.cost_per_power_beam_dump, float) and halhf1.cost_per_power_beam_dump > 0.0
    assert isinstance(halhf1.cost_per_power_reliquification_plant_nitrogen, float) and halhf1.cost_per_power_reliquification_plant_nitrogen > 0.0
    assert isinstance(halhf1.cost_per_source, float) and halhf1.cost_per_source > 0.0
    assert isinstance(halhf1.cost_per_source_polarized_electrons, float) and halhf1.cost_per_source_polarized_electrons > 0.0
    assert isinstance(halhf1.cost_per_source_polarized_positrons, float) and halhf1.cost_per_source_polarized_positrons > 0.0
    assert isinstance(halhf1.cost_snowmass_itf_accounting(), float) and halhf1.cost_snowmass_itf_accounting() > 0.0
    assert isinstance(halhf1.emissions_per_energy_usage, float) and halhf1.emissions_per_energy_usage > 0.0
    assert np.isclose(halhf1.energy_asymmetry, 4.0, rtol=1e-15, atol=0.0)
    assert isinstance(halhf1.energy_cost(), float) and halhf1.energy_cost() > 0.0
    assert isinstance(halhf1.energy_emissions(), float) and halhf1.energy_emissions() > 0.0
    assert isinstance(halhf1.energy_usage(), float) and halhf1.energy_usage() > 0.0
    assert np.isclose(halhf1.enhancement_factor(), 1.0, rtol=1e-15, atol=0.0)
    assert isinstance(halhf1.full_luminosity(), float) and halhf1.full_luminosity() > 1e37
    assert isinstance(halhf1.full_luminosity_per_crossing(), float) and halhf1.full_luminosity_per_crossing() > 1e33
    assert isinstance(halhf1.full_luminosity_per_power(), float) and halhf1.full_luminosity_per_power() > 1e29
    assert isinstance(halhf1.full_programme_cost(), float) and halhf1.full_programme_cost() > 0.0
    assert isinstance(halhf1.geometric_luminosity_per_crossing(), float) and halhf1.geometric_luminosity_per_crossing() > 0.0
    assert np.isclose(halhf1.get_com_energy(), 250e9, rtol=1e-15, atol=0.0)
    assert isinstance(halhf1.get_cost(), float) and halhf1.get_cost() > 0.0
    assert np.isclose(halhf1.get_energy_asymmetry(), 4.0, rtol=1e-15, atol=0.0)
    assert isinstance(halhf1.integrated_energy_usage(), float) and halhf1.integrated_energy_usage() > 0.0
    assert isinstance(halhf1.integrated_runtime(), float) and halhf1.integrated_runtime() > 0.0
    assert halhf1.is_scan() is False
    assert isinstance(halhf1.learning_curve_klystrons, float) and halhf1.learning_curve_klystrons >= 0.0
    assert isinstance(halhf1.length_end_to_end(), float) and halhf1.length_end_to_end() > 0.0
    assert isinstance(halhf1.maintenance_cost(), float) and halhf1.maintenance_cost() > 0.0
    assert isinstance(halhf1.maintenance_labor(), float) and halhf1.maintenance_labor() > 0.0
    assert isinstance(halhf1.maintenance_labor_per_construction_cost, float) and halhf1.maintenance_labor_per_construction_cost > 0.0
    #halhf1.maximum_upsilon()  # TODO: why does this give FALSE?
    assert isinstance(halhf1.num_bunches_in_train, int) and halhf1.num_bunches_in_train > 0
    #halhf1.num_coherent_pairs()  # TODO: why does this give FALSE?
    #halhf1.num_photons_beam1()  # TODO: why does this give FALSE?
    #halhf1.num_photons_beam2()  # TODO: why does this give FALSE?
    assert halhf1.num_shots == 1
    assert halhf1.num_shots_per_step == 1
    assert halhf1.num_steps == 1
    assert isinstance(halhf1.overhead_cost(), float) and halhf1.overhead_cost() > 0.0
    assert isinstance(halhf1.overhead_cost_design_and_development(), float) and halhf1.overhead_cost_design_and_development() > 0.0
    assert isinstance(halhf1.overhead_cost_management_inspection(), float) and halhf1.overhead_cost_management_inspection() > 0.0
    assert isinstance(halhf1.peak_luminosity(), float) and halhf1.peak_luminosity() > 0.0
    assert isinstance(halhf1.peak_luminosity_per_crossing(), float) and halhf1.peak_luminosity_per_crossing() > 0.0
    assert isinstance(halhf1.peak_luminosity_per_power(), float) and halhf1.peak_luminosity_per_power() > 0.0
    assert isinstance(halhf1.power_overhead(), float) and halhf1.power_overhead() > 0.0
    assert isinstance(halhf1.programme_duration(), float) and halhf1.programme_duration() > 0.0
    assert isinstance(halhf1.rep_rate_trains, float) and halhf1.rep_rate_trains > 0.0
    halhf1.run_name == 'test_HALHFv1'
    assert isinstance(halhf1.target_integrated_luminosity, float) and halhf1.target_integrated_luminosity > 0.0
    assert isinstance(halhf1.target_integrated_luminosity_250GeV, float) and halhf1.target_integrated_luminosity_250GeV > 0.0
    assert isinstance(halhf1.target_integrated_luminosity_550GeV, float) and halhf1.target_integrated_luminosity_550GeV > 0.0
    assert isinstance(halhf1.total_emissions(), float) and halhf1.total_emissions() > 0.0
    assert isinstance(halhf1.total_tunnel_length(), float) and halhf1.total_tunnel_length() > 0.0
    assert isinstance(halhf1.uptime_percentage, float) and halhf1.uptime_percentage > 0.0 and halhf1.uptime_percentage < 1.0
    assert isinstance(halhf1.wallplug_power(), float) and halhf1.wallplug_power() > 0.0

    #halhf1.final_beam

    # Test plotting and printing
    plt.ion()
    halhf1.plot_survey()
    
    halhf1.print_cost()
    halhf1.print_emissions()
    halhf1.print_power()

    # Remove output directory
    shutil.rmtree(halhf1.run_path())


@pytest.mark.presets
def test_HALHFv2():
    """
    Test for the HALHFv2 collider preset.
    """

    np.random.seed(42)

    halhf2 = HALHFv2()
    halhf2.run('test_HALHFv2', overwrite=True, verbose=False)

    # Tests
    assert isinstance(halhf2.bunch_separation, float) and halhf2.bunch_separation > 0.0
    assert isinstance(halhf2.carbon_tax_cost(), float) and halhf2.carbon_tax_cost() > 0.0
    assert isinstance(halhf2.collision_rate(), float) and halhf2.collision_rate() > 0.0
    assert np.isclose(halhf2.com_energy, 250e9, rtol=1e-15, atol=0.0)
    assert isinstance(halhf2.construction_cost(), float) and halhf2.construction_cost() > 0.0
    assert isinstance(halhf2.construction_emissions(), float) and halhf2.construction_emissions() > 0.0
    assert isinstance(halhf2.cost_carbon_tax_per_emissions, float) and halhf2.cost_carbon_tax_per_emissions > 0.0
    assert isinstance(halhf2.cost_eu_accounting(), float) and halhf2.cost_eu_accounting() > 0.0
    assert isinstance(halhf2.cost_factor_infrastructure_and_services, float) and halhf2.cost_factor_infrastructure_and_services > 0.0
    assert isinstance(halhf2.cost_per_energy, float) and halhf2.cost_per_energy > 0.0
    assert isinstance(halhf2.cost_per_ip, float) and halhf2.cost_per_ip > 0.0
    assert isinstance(halhf2.cost_per_length_bds, float) and halhf2.cost_per_length_bds > 0.0
    assert isinstance(halhf2.cost_per_length_combiner_ring, float) and halhf2.cost_per_length_combiner_ring > 0.0
    assert isinstance(halhf2.cost_per_length_damping_ring, float) and halhf2.cost_per_length_damping_ring > 0.0
    assert isinstance(halhf2.cost_per_length_interstage, float) and halhf2.cost_per_length_interstage > 0.0
    assert isinstance(halhf2.cost_per_length_plasma_stage, float) and halhf2.cost_per_length_plasma_stage > 0.0
    assert isinstance(halhf2.cost_per_length_rf_structure_normalconducting, float) and halhf2.cost_per_length_rf_structure_normalconducting > 0.0
    assert isinstance(halhf2.cost_per_length_rf_structure_superconducting, float) and halhf2.cost_per_length_rf_structure_superconducting > 0.0
    assert isinstance(halhf2.cost_per_length_surfacebuilding, float) and halhf2.cost_per_length_surfacebuilding > 0.0
    assert isinstance(halhf2.cost_per_length_transfer_line, float) and halhf2.cost_per_length_transfer_line > 0.0
    assert isinstance(halhf2.cost_per_length_tunnel_large, float) and halhf2.cost_per_length_tunnel_large > 0.0
    assert isinstance(halhf2.cost_per_length_turnaround, float) and halhf2.cost_per_length_turnaround > 0.0
    assert isinstance(halhf2.cost_per_power_beam_dump, float) and halhf2.cost_per_power_beam_dump > 0.0
    assert isinstance(halhf2.cost_per_power_reliquification_plant_nitrogen, float) and halhf2.cost_per_power_reliquification_plant_nitrogen > 0.0
    assert isinstance(halhf2.cost_per_source, float) and halhf2.cost_per_source > 0.0
    assert isinstance(halhf2.cost_per_source_polarized_electrons, float) and halhf2.cost_per_source_polarized_electrons > 0.0
    assert isinstance(halhf2.cost_per_source_polarized_positrons, float) and halhf2.cost_per_source_polarized_positrons > 0.0
    assert isinstance(halhf2.cost_snowmass_itf_accounting(), float) and halhf2.cost_snowmass_itf_accounting() > 0.0
    assert isinstance(halhf2.emissions_per_energy_usage, float) and halhf2.emissions_per_energy_usage > 0.0
    assert np.isclose(halhf2.energy_asymmetry, 3.0, rtol=1e-15, atol=0.0)
    assert isinstance(halhf2.energy_cost(), float) and halhf2.energy_cost() > 0.0
    assert isinstance(halhf2.energy_emissions(), float) and halhf2.energy_emissions() > 0.0
    assert isinstance(halhf2.energy_usage(), float) and halhf2.energy_usage() > 0.0
    assert np.isclose(halhf2.enhancement_factor(), 1.0, rtol=1e-15, atol=0.0)
    assert isinstance(halhf2.full_luminosity(), float) and halhf2.full_luminosity() > 1e37
    assert isinstance(halhf2.full_luminosity_per_crossing(), float) and halhf2.full_luminosity_per_crossing() > 1e33
    assert isinstance(halhf2.full_luminosity_per_power(), float) and halhf2.full_luminosity_per_power() > 1e29
    assert isinstance(halhf2.full_programme_cost(), float) and halhf2.full_programme_cost() > 0.0
    assert isinstance(halhf2.geometric_luminosity_per_crossing(), float) and halhf2.geometric_luminosity_per_crossing() > 0.0
    assert np.isclose(halhf2.get_com_energy(), 250e9, rtol=1e-15, atol=0.0)
    assert isinstance(halhf2.get_cost(), float) and halhf2.get_cost() > 0.0
    assert np.isclose(halhf2.get_energy_asymmetry(), 3.0, rtol=1e-15, atol=0.0)
    assert isinstance(halhf2.integrated_energy_usage(), float) and halhf2.integrated_energy_usage() > 0.0
    assert isinstance(halhf2.integrated_runtime(), float) and halhf2.integrated_runtime() > 0.0
    assert halhf2.is_scan() is False
    assert isinstance(halhf2.learning_curve_klystrons, float) and halhf2.learning_curve_klystrons >= 0.0
    assert isinstance(halhf2.length_end_to_end(), float) and halhf2.length_end_to_end() > 0.0
    assert isinstance(halhf2.maintenance_cost(), float) and halhf2.maintenance_cost() > 0.0
    assert isinstance(halhf2.maintenance_labor(), float) and halhf2.maintenance_labor() > 0.0
    assert isinstance(halhf2.maintenance_labor_per_construction_cost, float) and halhf2.maintenance_labor_per_construction_cost > 0.0
    #halhf2.maximum_upsilon()  # TODO: why does this give FALSE?
    assert isinstance(halhf2.num_bunches_in_train, int) and halhf2.num_bunches_in_train > 0
    #halhf2.num_coherent_pairs()  # TODO: why does this give FALSE?
    #halhf2.num_photons_beam1()  # TODO: why does this give FALSE?
    #halhf2.num_photons_beam2()  # TODO: why does this give FALSE?
    assert halhf2.num_shots == 1
    assert halhf2.num_shots_per_step == 1
    assert halhf2.num_steps == 1
    assert isinstance(halhf2.overhead_cost(), float) and halhf2.overhead_cost() > 0.0
    assert isinstance(halhf2.overhead_cost_design_and_development(), float) and halhf2.overhead_cost_design_and_development() > 0.0
    assert isinstance(halhf2.overhead_cost_management_inspection(), float) and halhf2.overhead_cost_management_inspection() > 0.0
    assert isinstance(halhf2.peak_luminosity(), float) and halhf2.peak_luminosity() > 0.0
    assert isinstance(halhf2.peak_luminosity_per_crossing(), float) and halhf2.peak_luminosity_per_crossing() > 0.0
    assert isinstance(halhf2.peak_luminosity_per_power(), float) and halhf2.peak_luminosity_per_power() > 0.0
    assert isinstance(halhf2.power_overhead(), float) and halhf2.power_overhead() > 0.0
    assert isinstance(halhf2.programme_duration(), float) and halhf2.programme_duration() > 0.0
    assert isinstance(halhf2.rep_rate_trains, float) and halhf2.rep_rate_trains > 0.0
    halhf2.run_name == 'test_HALHFv2'
    assert isinstance(halhf2.target_integrated_luminosity, float) and halhf2.target_integrated_luminosity > 0.0
    assert isinstance(halhf2.target_integrated_luminosity_250GeV, float) and halhf2.target_integrated_luminosity_250GeV > 0.0
    assert isinstance(halhf2.target_integrated_luminosity_550GeV, float) and halhf2.target_integrated_luminosity_550GeV > 0.0
    assert isinstance(halhf2.total_emissions(), float) and halhf2.total_emissions() > 0.0
    assert isinstance(halhf2.total_tunnel_length(), float) and halhf2.total_tunnel_length() > 0.0
    assert isinstance(halhf2.uptime_percentage, float) and halhf2.uptime_percentage > 0.0 and halhf2.uptime_percentage < 1.0
    assert isinstance(halhf2.wallplug_power(), float) and halhf2.wallplug_power() > 0.0

    #halhf2.final_beam
    
    # Test plotting and printing
    plt.ion()
    halhf2.plot_survey()
    halhf2.plot_luminosity_per_power()
    halhf2.plot_luminosity()
    
    halhf2.print_cost()
    halhf2.print_emissions()
    halhf2.print_power()

    # Remove output directory
    shutil.rmtree(halhf2.run_path())