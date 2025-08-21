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

    my_rtol = 1e-5  # Standard relative tolerance

    # Tests
    assert np.isclose(cool_copper_collider.bunch_charge, 1e-9, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.bunch_separation, 5.2600000000000004e-09, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.bunch_separation_ns, 5.26, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.carbon_tax_cost(), 199737857.5049092, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.collision_rate(), 15960.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.com_energy, 250e9, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.construction_cost(), 3412578205.4742675, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.construction_emissions(), 85472.1455685134, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_carbon_tax_per_emissions, 800, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_eu_accounting(), 3412578205.4742675, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_factor_infrastructure_and_services, 0.13530999999999999, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_energy, 1.388888888888889e-08, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_ip, 76762820.51282051, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_bds, 40440.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_combiner_ring, 79000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_damping_ring, 260000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_interstage, 40440.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_plasma_stage, 46200.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_rf_structure_normalconducting, 115000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_rf_structure_superconducting, 106000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_surfacebuilding, 33258.01282051282, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_transfer_line, 15400.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_tunnel_large, 41207.961149999996, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_source, 10000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_source_polarized_electrons, 82000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_source_polarized_positrons, 178000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_snowmass_itf_accounting(), 4163345410.678606, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.emissions_per_energy_usage, 5.5555555555555555e-12, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.energy_asymmetry, 1.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.energy_cost(), 410500440.78155774, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.energy_emissions(), 164200.17631262308, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.energy_usage(), 6960.781269490861, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.enhancement_factor(), 1.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.full_luminosity(), 7.517522753606623e+37, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.full_luminosity_per_crossing(), 4.7102272892272076e+33, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.full_luminosity_per_power(), 6.76680827062969e+29, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.full_programme_cost(), 5061476808.894532, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.geometric_luminosity_per_crossing(), 4.7102272892272076e+33, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.get_com_energy(), 250e9, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.get_cost(), 5103156308.894532, rtol=my_rtol, atol=0.0)
    #cool_copper_collider.get_cost_breakdown()
    #cool_copper_collider.get_cost_breakdown_overheads()
    #cool_copper_collider.get_cost_civil_construction()
    #event = cool_copper_collider.get_event()
    assert np.isclose(cool_copper_collider.get_energy_asymmetry(), 1.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.integrated_energy_usage(), 2.9556031736272156e+16, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.integrated_runtime(), 266045087.6640813, rtol=my_rtol, atol=0.0)
    assert cool_copper_collider.is_scan() is False
    assert np.isclose(cool_copper_collider.learning_curve_klystrons, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.length_end_to_end(), 6374.546934897977, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.linac_gradient, 70000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.maintenance_cost(), 287893096.0920645, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.maintenance_labor(), 341.2578205474267, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.maintenance_labor_per_construction_cost, 1e-7, rtol=1e-15, atol=0.0)
    #cool_copper_collider.maximum_upsilon()  # TODO: why does this give FALSE?
    assert cool_copper_collider.num_bunches_in_train == 133
    #cool_copper_collider.num_coherent_pairs()  # TODO: why does this give FALSE?
    #cool_copper_collider.num_photons_beam1()  # TODO: why does this give FALSE?
    #cool_copper_collider.num_photons_beam2()  # TODO: why does this give FALSE?
    assert cool_copper_collider.num_shots == 1
    assert cool_copper_collider.num_shots_per_step == 1
    assert cool_copper_collider.num_steps == 1
    assert cool_copper_collider.num_structures_per_klystron == 130
    assert np.isclose(cool_copper_collider.overhead_cost(), 750767205.2043388, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.overhead_cost_design_and_development(), 341257820.54742676, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.overhead_cost_management_inspection(), 409509384.6569121, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.peak_luminosity(), 7.517522400334487e+37, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.peak_luminosity_per_crossing(), 4.7102270678787516e+33, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.peak_luminosity_per_power(), 6.76680827062969e+29, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.power_overhead(), 22218813.812214825, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.programme_duration(), 380064428.80911547, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.rep_rate_trains, 120.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.rf_frequency, 5712000000.0, rtol=1e-15, atol=0.0)
    cool_copper_collider.run_name == 'test_C3'
    assert np.isclose(cool_copper_collider.target_integrated_luminosity, 2e46, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.total_emissions(), 249672.32188113648, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.total_tunnel_length(), 7743.865907597206, rtol=my_rtol, atol=0.0)
    assert np.isclose(cool_copper_collider.uptime_percentage, 0.7, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.wallplug_power(), 111094069.06107412, rtol=my_rtol, atol=0.0)

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

    my_rtol = 1e-5  # Standard relative tolerance

    # Tests
    assert np.isclose(halhf1.bunch_separation, 8e-8, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.carbon_tax_cost(), 256498153.59937534, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.collision_rate(), 10000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.com_energy, 250e9, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.construction_cost(), 2513208775.9133253, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.construction_emissions(), 154323.74733138402, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.cost_carbon_tax_per_emissions, 800, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_eu_accounting(), 2513208775.9133253, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.cost_factor_infrastructure_and_services, 0.13530999999999999, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.cost_per_energy, 1.388888888888889e-08, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.cost_per_ip, 76762820.51282051, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.cost_per_length_bds, 40440.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_per_length_combiner_ring, 79000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_per_length_damping_ring, 260000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_per_length_interstage, 40440.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_per_length_plasma_stage, 46200.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_per_length_rf_structure_normalconducting, 115000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_per_length_rf_structure_superconducting, 106000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_per_length_surfacebuilding, 33258.01282051282, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.cost_per_length_transfer_line, 15400.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_per_length_tunnel_large, 41207.961149999996, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.cost_per_length_turnaround, 40440.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_per_power_beam_dump, 2.392857142857143, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.cost_per_power_reliquification_plant_nitrogen, 13.5, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_per_source, 10000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_per_source_polarized_electrons, 82000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_per_source_polarized_positrons, 178000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.cost_snowmass_itf_accounting(), 3066114706.614257, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.emissions_per_energy_usage, 5.5555555555555555e-12, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.energy_asymmetry, 4.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.energy_cost(), 415747361.6695879, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.energy_emissions(), 166298.944667835178, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.energy_usage(), 8596.006540508495, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.enhancement_factor(), 1.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.full_luminosity(), 5.743342747856962e+37, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.full_luminosity_per_crossing(), 5.743342747856962e+33, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.full_luminosity_per_power(), 6.681408071052044e+29, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.full_programme_cost(), 4015875707.1839, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.geometric_luminosity_per_crossing(), 5.743342747856962e+33, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.get_com_energy(), 250e9, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.get_cost(), 4075001830.951311, rtol=my_rtol, atol=0.0)
    
    assert np.isclose(halhf1.get_energy_asymmetry(), 4.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.integrated_energy_usage(), 2.993381004021033e+16, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.integrated_runtime(), 348229260.86837995, rtol=my_rtol, atol=0.0)
    assert halhf1.is_scan() is False
    assert np.isclose(halhf1.learning_curve_klystrons, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.length_end_to_end(), 3995.9128827084896, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.maintenance_cost(), 277515485.3006796, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.maintenance_labor(), 251.32087759133253, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.maintenance_labor_per_construction_cost, 1e-7, rtol=1e-15, atol=0.0)
    #halhf1.maximum_upsilon()  # TODO: why does this give FALSE?
    assert halhf1.num_bunches_in_train == 100
    # #halhf1.num_coherent_pairs()  # TODO: why does this give FALSE?
    # #halhf1.num_photons_beam1()  # TODO: why does this give FALSE?
    # #halhf1.num_photons_beam2()  # TODO: why does this give FALSE?
    assert halhf1.num_shots == 1
    assert halhf1.num_shots_per_step == 1
    assert halhf1.num_steps == 1
    assert np.isclose(halhf1.overhead_cost(), 552905930.7009315, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.overhead_cost_design_and_development(), 251320877.59133255, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.overhead_cost_management_inspection(), 301585053.10959905, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.peak_luminosity(), 5.743342747856962e+37, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.peak_luminosity_per_crossing(), 5.743342747856962e+33, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.peak_luminosity_per_power(), 6.681408071052044e+29, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.power_overhead(), 17192013.081016988, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.programme_duration(), 497470372.66911423, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.rep_rate_trains, 100.0, rtol=1e-15, atol=0.0)
    assert halhf1.run_name == 'test_HALHFv1'
    assert np.isclose(halhf1.target_integrated_luminosity, 2e46, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.target_integrated_luminosity_250GeV, 2e46, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.target_integrated_luminosity_550GeV, 4e46, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.total_emissions(), 320622.6919992192, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.total_tunnel_length(), 13981.893138908077, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf1.uptime_percentage, 0.7, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf1.wallplug_power(), 85960065.40508494, rtol=my_rtol, atol=0.0)

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

    my_rtol = 1e-5  # Standard relative tolerance

    # Tests
    assert np.isclose(halhf2.bunch_separation, 1.6e-8, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.carbon_tax_cost(), 228303479.8060721, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.collision_rate(), 16000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.com_energy, 250e9, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.construction_cost(), 2906407959.1288686, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.construction_emissions(), 157057.12228970297, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.cost_carbon_tax_per_emissions, 800, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_eu_accounting(), 2906407959.1288686, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.cost_factor_infrastructure_and_services, 0.13530999999999999, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.cost_per_energy, 1.388888888888889e-08, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.cost_per_ip, 76762820.51282051, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.cost_per_length_bds, 40440.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_per_length_combiner_ring, 79000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_per_length_damping_ring, 260000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_per_length_interstage, 40440.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_per_length_plasma_stage, 46200.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_per_length_rf_structure_normalconducting, 115000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_per_length_rf_structure_superconducting, 106000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_per_length_surfacebuilding, 33258.01282051282, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.cost_per_length_transfer_line, 15400.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_per_length_tunnel_large, 41207.961149999996, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.cost_per_length_turnaround, 40440.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_per_power_beam_dump, 2.392857142857143, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.cost_per_power_reliquification_plant_nitrogen, 13.5, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_per_source, 10000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_per_source_polarized_electrons, 82000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_per_source_polarized_positrons, 178000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.cost_snowmass_itf_accounting(), 3545817710.1372194, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.emissions_per_energy_usage, 5.5555555555555555e-12, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.energy_asymmetry, 3.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.energy_cost(), 320805568.66971785, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.energy_emissions(), 128322.22746788713, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.energy_usage(), 6631.594455276813, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.enhancement_factor(), 1.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.full_luminosity(), 9.187419425660911e+37, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.full_luminosity_per_crossing(), 5.74213714103807e+33, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.full_luminosity_per_power(), 8.658757980094825e+29, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.full_programme_cost(), 4295552465.998911, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.geometric_luminosity_per_crossing(), 5.74213714103807e+33, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.get_com_energy(), 250e9, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.get_cost(), 4360200281.107528, rtol=my_rtol, atol=0.0)
    
    assert np.isclose(halhf2.get_energy_asymmetry(), 3.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.integrated_energy_usage(), 2.3098000944219684e+16, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.integrated_runtime(), 217688983.96147043, rtol=my_rtol, atol=0.0)
    assert halhf2.is_scan() is False
    assert np.isclose(halhf2.learning_curve_klystrons, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.length_end_to_end(), 4929.176582547312, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.maintenance_cost(), 200625636.794811, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.maintenance_labor(), 290.64079591288686, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.maintenance_labor_per_construction_cost, 1e-7, rtol=1e-15, atol=0.0)
    #halhf2.maximum_upsilon()  # TODO: why does this give FALSE?
    assert halhf2.num_bunches_in_train == 160
    # #halhf2.num_coherent_pairs()  # TODO: why does this give FALSE?
    # #halhf2.num_photons_beam1()  # TODO: why does this give FALSE?
    # #halhf2.num_photons_beam2()  # TODO: why does this give FALSE?
    assert halhf2.num_shots == 1
    assert halhf2.num_shots_per_step == 1
    assert halhf2.num_steps == 1
    assert np.isclose(halhf2.overhead_cost(), 639409751.0083511, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.overhead_cost_design_and_development(), 290640795.91288686, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.overhead_cost_management_inspection(), 348768955.0954642, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.peak_luminosity(), 9.187419425660911e+37, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.peak_luminosity_per_crossing(), 5.74213714103807e+33, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.peak_luminosity_per_power(), 8.658757980094825e+29, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.power_overhead(), 21221102.256885804, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.programme_duration(), 310984262.8021006, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.rep_rate_trains, 100.0, rtol=1e-15, atol=0.0)
    assert halhf2.run_name == 'test_HALHFv2'
    assert np.isclose(halhf2.target_integrated_luminosity, 2e46, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.target_integrated_luminosity_250GeV, 2e46, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.target_integrated_luminosity_550GeV, 4e46, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.total_emissions(), 285379.3497575901, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.total_tunnel_length(), 14229.539772926864, rtol=my_rtol, atol=0.0)
    assert np.isclose(halhf2.uptime_percentage, 0.7, rtol=1e-15, atol=0.0)
    assert np.isclose(halhf2.wallplug_power(), 106105511.28442901, rtol=my_rtol, atol=0.0)

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