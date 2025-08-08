# -*- coding: utf-8 -*-
"""
ABEL : Tests of collider presets
=======================================

This file is a part of ABEL.
Copyright 2022– C.A.Lindstrøm, B.Chen, O.G. Finnerud,
D. Kallvik, E. Hørlyk, K.N. Sjobak, E.Adli, University of Oslo

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
from abel import *
import shutil
#import numpy as np


@pytest.mark.presets
def test_C3():
    """
    Test for the C3 collider preset.
    """

    from abel import C3
    np.random.seed(42)

    cool_copper_collider = C3()
    cool_copper_collider.run('test_C3', overwrite=True)

    # Tests
    assert np.isclose(cool_copper_collider.bunch_charge, 1e-9, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.bunch_separation, 5.2600000000000004e-09, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.bunch_separation_ns, 5.26, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.carbon_tax_cost(), 199718369.41390473, rtol=1e-4, atol=0.0)
    assert np.isclose(cool_copper_collider.collision_rate(), 15960.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.com_energy, 250e9, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.construction_cost(), 3412578205.4742675, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.construction_emissions(), 85472.1455685134, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_carbon_tax_per_emissions, 800, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_eu_accounting(), 3412578205.4742675, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_factor_infrastructure_and_services, 0.13530999999999999, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_energy, 1.388888888888889e-08, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_ip, 76762820.51282051, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_bds, 40440.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_combiner_ring, 79000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_damping_ring, 260000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_interstage, 40440.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_plasma_stage, 46200.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_rf_structure_normalconducting, 115000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_rf_structure_superconducting, 106000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_surfacebuilding, 33258.01282051282, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_transfer_line, 15400.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_length_tunnel_large, 41207.961149999996, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_source, 10000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_source_polarized_electrons, 82000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_per_source_polarized_positrons, 178000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.cost_snowmass_itf_accounting(), 4163345410.678606, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.emissions_per_energy_usage, 5.5555555555555555e-12, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.energy_asymmetry, 1.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.energy_cost(), 410439540.49716884, rtol=1e-3, atol=0.0)
    assert np.isclose(cool_copper_collider.energy_emissions(), 164175.8161988675, rtol=1e-3, atol=0.0)
    assert np.isclose(cool_copper_collider.energy_usage(), 6959.748687219503, rtol=1e-3, atol=0.0)
    assert np.isclose(cool_copper_collider.enhancement_factor(), 1.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.full_luminosity(), 7.517522753606623e+37, rtol=1e-5, atol=0.0)
    assert np.isclose(cool_copper_collider.full_luminosity_per_crossing(), 4.7102272892272076e+33, rtol=1e-5, atol=0.0)
    assert np.isclose(cool_copper_collider.full_luminosity_per_power(), 6.767812317529234e+29, rtol=1e15, atol=0.0)
    assert np.isclose(cool_copper_collider.full_programme_cost(), 5061396416.681745, rtol=1e-3, atol=0.0)
    assert np.isclose(cool_copper_collider.geometric_luminosity_per_crossing(), 4.7102272892272076e+33, rtol=1e-5, atol=0.0)
    assert np.isclose(cool_copper_collider.get_com_energy(), 250e9, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.get_cost(), 5103075916.681745, rtol=1e-3, atol=0.0)
    #cool_copper_collider.get_cost_breakdown()
    #cool_copper_collider.get_cost_breakdown_overheads()
    #cool_copper_collider.get_cost_civil_construction()
    #event = cool_copper_collider.get_event()
    assert np.isclose(cool_copper_collider.get_energy_asymmetry(), 1.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.integrated_energy_usage(), 2.955164691579615e+16, rtol=1e-3, atol=0.0)
    assert np.isclose(cool_copper_collider.integrated_runtime(), 266045087.6640813, rtol=1e-5, atol=0.0)
    assert np.isclose(cool_copper_collider.is_scan(), False, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.learning_curve_klystrons, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.length_end_to_end(), 6374.546934897977, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.linac_gradient, 70000000.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.maintenance_cost(), 287893096.0920645, rtol=1e-5, atol=0.0)
    assert np.isclose(cool_copper_collider.maintenance_labor(), 341.2578205474267, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.maintenance_labor_per_construction_cost, 1e-7, rtol=1e-15, atol=0.0)
    #cool_copper_collider.maximum_upsilon()
    assert cool_copper_collider.num_bunches_in_train == 133
    #cool_copper_collider.num_coherent_pairs()
    #cool_copper_collider.num_photons_beam1()
    #cool_copper_collider.num_photons_beam2()
    assert cool_copper_collider.num_shots == 1
    assert cool_copper_collider.num_shots_per_step == 1
    assert cool_copper_collider.num_steps == 1
    assert cool_copper_collider.num_structures_per_klystron == 130
    assert np.isclose(cool_copper_collider.overhead_cost(), 750767205.2043388, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.overhead_cost_design_and_development(), 341257820.54742676, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.overhead_cost_management_inspection(), 409509384.6569121, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.peak_luminosity(), 7.517522400334487e+37, rtol=1e-5, atol=0.0)
    assert np.isclose(cool_copper_collider.peak_luminosity_per_crossing(), 4.7102270678787516e+33, rtol=1e-5, atol=0.0)
    assert np.isclose(cool_copper_collider.peak_luminosity_per_power(), 6.767112352049366e+29, rtol=1e-3, atol=0.0)
    assert np.isclose(cool_copper_collider.power_overhead(), 22217814.65785141, rtol=1e-3, atol=0.0)
    assert np.isclose(cool_copper_collider.programme_duration(), 380064428.80911547, rtol=1e-5, atol=0.0)
    assert np.isclose(cool_copper_collider.rep_rate_trains, 120.0, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.rf_frequency, 5712000000.0, rtol=1e-15, atol=0.0)
    cool_copper_collider.run_name == 'test_C3'
    assert np.isclose(cool_copper_collider.target_integrated_luminosity, 2e46, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.total_emissions(), 249664.94351646234, rtol=1e-3, atol=0.0)
    assert np.isclose(cool_copper_collider.total_tunnel_length(), 7743.865907597206, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.uptime_percentage, 0.7, rtol=1e-15, atol=0.0)
    assert np.isclose(cool_copper_collider.wallplug_power(), 111089073.28925705, rtol=1e-3, atol=0.0)

    #cool_copper_collider.print_emissions()

    #cool_copper_collider.final_beam

    #cool_copper_collider.plot_survey()
    #cool_copper_collider.print_cost()
    #cool_copper_collider.print_power()

    # Remove output directory
    shutil.rmtree(cool_copper_collider.run_path())