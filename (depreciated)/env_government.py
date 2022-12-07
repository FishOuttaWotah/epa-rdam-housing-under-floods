"""
overlord collects data in ledgers
also overlaps with datacollection
rename most likely needed

- probably the creation of some routine that would change the ledger after agent run?
- inactive agents gets superagent-run to update changes and stuff
"""
from typing import TYPE_CHECKING

import agent_base

## includes a function to derive the market share of firms. This could be done via numpy array methods?

class Government(agent_base.CustomAgent):
    # TODO: fill up government stagelist when defined
    stagelist = []

    def __init__(self, unique_id, model):

        super(Government, self).__init__(unique_id=unique_id, model=model)

        # -- General government attributes -- #
        # --------
        # COMMENT: change list for variable saving structure
        # items disabled: region
        # --------
        self.type = "Gov"
        self.house_quarter_income_ratio = None
        self.total_savings_pre_flood = None
        self.repair_exp = None
        self.consumption = None
        self.monetary_damage = None
        self.total_damage = None
        self.flooded = None
        self.at_risk = None
        self.worry = None
        self.risk_perc = None
        self.SE_wet = None
        self.RE_wet = None
        self.height = 0
        # self.region = region  # disabled
        self.net_worth = None

        # -- Subsidies -- #
        self.fiscal_balance = 0
        self.fiscal_revenues = 0
        self.unempl_subsidy_frac = 1
        self.unempl_subsidy = [0, 0]
        self.unemployment_expenditure = [0, 0]
        self.min_wage = [1, 1]
        self.min_wage_frac = 0.5
        self.wage = None

        self.market_shares_ids_cons_region0 = []
        self.market_shares_ids_cons_region1 = []
        self.market_shares_normalized = 0
        self.market_shares_normalized_cap = 0
        self.average_normalized_market_share = 0
        self.market_shares_normalized_serv = 0
        self.average_normalized_comp = [0, 0, 0]
        self.serv_average_normalized_comp = [0, 0, 0]
        self.norm_price_unfilled_demand = 0
        self.norm_price_unfilled_demand_serv = 0

        self.average_wages = [0, 0, 0, 0]
        self.net_sales_cons_firms = [0, 0, 0, 0]
        self.salaries_cap = [1, 1]
        self.salaries_cons = [1, 1]
        self.salaries_serv = [1, 1]
        self.av_price_cons = [1, 1]
        self.av_price_cap = [1, 1]
        self.av_price_serv = [1, 1]
        self.av_price = [1.15, 1.15]
        self.transport_cost = self.model.transport_cost
        self.transport_cost_RoW = self.transport_cost * 2
        self.aggregate_cons = [0, 0, 0, 0]
        self.aggregate_serv = [0, 0, 0, 0]
        self.regional_pop_hous = [self.model.num_households / 2,
                                  self.model.num_households / 2]
        self.firms_pop = [self.model.num_firms1 / 2,
                          self.model.num_firms1 / 2,
                          self.model.num_firms2 / 2,
                          self.model.num_firms2 / 2,
                          self.model.num_firms3 / 2,
                          self.model.num_firms3 / 2]
        self.aggregate_unemployment = [0, 0]
        self.aggregate_employment = [0, 0]
        self.regional_av_prod = [1, 1, 0, 0]
        self.regional_av_prod_cons = [1, 1, 0, 0]
        self.regional_av_prod_cap = [1, 1, 0, 0]
        self.regional_av_prod_serv = [1, 1, 0, 0]
        self.prev_unemployment_rates = [0, 0, 0, 0, 0]
        self.unemployment_rates = [0, 0, 0, 0, 0]
        self.cap_av_prod = [0, 0, 0, 0]
        self.orders = 0
        self.real_demands = [0, 0, 0]
        self.total_productions = [0, 0, 0, 0, 0, 0]
        self.export_demand = self.model.Exp_RoW
        self.export_demand_list = []
        self.fraction_exp = 0.004
        self.best_firm1 = [0, 0]
        self.serv_firm_out = 0
        self.cons_firm_out = 0
        self.av_size = [0, 0]
        self.av_net_worth = [0, 0]
        # self.av_migrant = 0
        # self.av_migrant_nw = [0, 0]
        self.firm_cap_out = 0
        self.tax_revenues = [0, 0]
        self.unemployment_cost = [0, 0]
        self.tax_rate = 0.25

        self.labour_demands = [1, 1, 1, 1, 1, 1, 1, 1]
        self.damage_coeff = 0
        # self.flood = False
        self.total_offers = [0, 0, 0, 0]
        self.feasible_productions = [0, 0, 0, 0]
        self.desired_productions = [0, 0, 0, 0]

        self.profits_firms = [0, 0, 0, 0, 0, 0]
        self.old_profits_firms = self.profits_firms
        self.entrants_cap_coastal = 0
        self.entrants_cap_inland = 0
        self.profitability_cons_coastal = 0
        self.profitability_cons_inland = 0

        self.entrants_cons_coastal = 0
        self.entrants_cons_inland = 0
        self.profitability_serv_coastal = 0
        self.profitability_serv_inland = 0
        self.entrants_serv_coastal = 0
        self.entrants_serv_inland = 0

        self.test_cons = [0, 0]

        # -- CCA parameters -- #
        self.CCAs = [0, 0, 0, 0, 0]
        self.recovery_time = None
        self.renounced_cons = None

        # -- Households Income parameters -- #
        self.income_pp = 0
        self.income_pp_change = [0, 0]
        self.previous_income_pp = 0

    # def normalization(self, data):
    #     """Normalize input data. """
    #     data_array = np.array(data)
    #     data_norm = ((data_array - data_array.min()) /
    #                  (data_array.max() - data_array.min()))
    #     return data_norm

    def minimum_wage(self):
        """Calculates minimum wages and unemployment subsidy. """
        # --------
        # COMMENT: checked values, nothing changed (1/4/2022)
        # --------
        fraction = self.min_wage_frac

        # Set minimum wage to fraction of average wage.
        self.min_wage = np.maximum(0.1,
                                   np.array(self.average_wages)[:2] *
                                   self.min_wage_frac)
        self.unempl_subsidy = np.around(np.maximum(0.1,
                                                   self.min_wage *
                                                   self.unempl_subsidy_frac), 3)

    def open_vacancies_list(self):
        """Posts job vacancies on the labour market. """
        # --------
        # COMMENT: values have been checked, did not change in new version
        #          (1/4/2022)
        # --------

        firms = self.model.firms_1_2

        # Collect the firms with vacancies in this region
        self.open_vacancies_all = [firm for firm in firms
                                   if (firm.open_vacancies
                                       and firm.region == self.region)]

        # # Get migrating firms
        # # --------
        # # COMMENT: currently not used
        # # COMMENT: total_size_migration == total_migrants ??
        # # --------
        # firm_migrants = [firm for firm in firms if firm.migration]
        # total_size_migration = sum(firm.migration for firm in firms)
        # total_nw_migration = sum(firm.net_worth for firm in firm_migrants)
        # total_migrants = len(firm_migrants)
        # if total_migrants > 0:
        #     self.av_migrant = total_size_migration / total_migrants
        #     self.av_migrant_nw = total_nw_migration / total_migrants
        # else:
        #     self.av_migrant = 0

    # def collect_taxes(self):
    #     """Collect taxes. """
    #     self.fiscal_revenues = 0
    #     pos = (0, self.region)
    #     regional_population = self.model.grid.get_cell_list_contents([pos])
    #     for i in range(len(regional_population)):
    #         if (regional_population[i].type == "Cons" or
    #             regional_population[i].type == "Cap" and
    #             regional_population[i].profits > 0):
    #             self.fiscal_revenues += (self.tax_rate *
    #                                      regional_population[i].profits)

    # ------------------------------------------------------------------------
    #                         NORMALIZATION FUNCTIONS
    # ------------------------------------------------------------------------
    # COMMENT: check if possible to combine all these functions
    #          into single function?
    # --------
    def norm_market_share(self, firms):
        """Normalization of market shares.

        Args:
            firms           : List of firm objects

        Returns:
            MS_norm_avg     : Average market share per region
            norm_ms_dict    : Dict of normalized market shares per firm
        """
        # --------
        # COMMENT: values have been checked, did not change in new version
        #          (1/4/2022)
        # --------

        # Normalize market shares for each market (Coastal, Inland, Export)
        MS = np.array([firm.market_share for firm in firms])
        MS_norm = np.around(MS / (np.linalg.norm(MS, 1, axis=0) + 1e-8), 8)

        # Get average normalized market share for each region
        MS_norm_avg = np.around(np.mean(MS_norm, axis=0), 8)

        # Get normalized regional market shares per firm
        firm_ids = [firm.unique_id for firm in firms]
        norm_ms_dict = dict(zip(firm_ids, MS_norm))

        return MS_norm_avg, norm_ms_dict

    def norm_market_share_cap(self, firms):
        """Normalize market share for capital good firms.

        Args:
            firms               : List of CapitalGoodFirm objects
        """
        # --------
        # COMMENT: values have been checked, did not change in new version
        #          (1/4/2022)
        # COMMENT: check why this is demand instead of market share ?
        # --------
        # Get demand for each market (Coastal, Inland) and normalize
        MS = np.array([firm.real_demand_cap for firm in firms])
        MS_norm = np.around(MS / (np.linalg.norm(MS, 1, axis=0) + 1e-8), 8)

        # Save normalized demand per firm
        firm_ids = [firm.unique_id for firm in firms]
        self.market_shares_normalized_cap = dict(zip(firm_ids, MS_norm))

    def norm_competitiveness(self, firms):
        """Normalize competitiveness.

        Args:
            firms               : List of firm objects

        Returns:
            norm_comp_dict      : Dict of normalized market shares per firm
            comp_norm_avg       : Average market share per region
        """
        # --------
        # COMMENT: values have been checked, did not change in new version
        #          (4/4/2022)
        # --------
        # Get competitiveness for all firms
        comp_all = [firm.competitiveness for firm in firms]
        # Convert to positive values
        comp_all_pos = comp_all + np.abs(np.min(comp_all, axis=0))
        # COMMENT: when adding noise here, this is summed in next step.
        #          Check if this should be changed?
        comp_all_pos += 1e-8
        # Normalize competitiveness and save per firm
        # COMMENT: rounding needed?
        norm_comp = np.around(comp_all_pos /
                              np.linalg.norm(comp_all_pos, 1, axis=0), 8)
        firm_ids = [firm.unique_id for firm in firms]
        norm_comp_dict = dict(zip(firm_ids, norm_comp))

        # Compute average competitiveness per region,
        # weighted by past market share
        # COMMENT: rounding needed?
        MS = [firm.market_share for firm in firms]
        comp_norm_avg = np.around(np.sum(np.multiply(norm_comp, MS),
                                         axis=0), 8)

        return norm_comp_dict, comp_norm_avg

    def norm_price_unf_demand(self, firms, firms_ids):
        """Normalize prices and unfilled demand.

        Args:
            firms           : List of firm objects

        Returns:
            norm_ms_dict    : Dict of normalized market shares per firm
        """
        # --------
        # COMMENT: values have been checked, did not change in new version
        #          (4/4/2022)
        # --------

        # Get prices and unfilled demand for all firms
        # COMMENT: again, check if noise adding should be done in next step
        prices = [firm.price for firm in firms]
        unfilled_demands = [firm.unfilled_demand for firm in firms]
        prices_unf_demands = np.column_stack((prices,
                                              unfilled_demands)) + 1e-8

        # Save normalized prices and unfilled demand per firm
        norm_prices_unf_demand = np.around(prices_unf_demands /
                                           np.linalg.norm(prices_unf_demands,
                                                          1, axis=0), 8)
        firm_ids = [firm.unique_id for firm in firms]
        norm_ms_dict = dict(zip(firm_ids, norm_prices_unf_demand))

        return norm_ms_dict

    def get_best_cap(self):
        """Get CapitalGoodFirm with best productivity/price ratio. """
        # --------
        # COMMENT: changed on 4/4/2022, values have been checked.
        # --------
        regions = [0, 1]
        for r in regions:
            # Get prod/price ratio for all Capital firms in this region
            firm_prod_dict = {firm: firm.productivity[0] / firm.price
                              for firm in self.model.firms1
                              if firm.region == r}
            # # --------
            # # COMMENT: should random sample still be included?
            # # --------
            # n_samples = math.ceil(len(firm_prod_dict)/5)
            # firm_prod_dict = dict(random.sample(firm_prod_dict.items(),
            #                                     n_samples))

            # Get firm with best prod/price ratio
            self.best_firm1[r] = max(firm_prod_dict, key=firm_prod_dict.get,
                                     default=None)

    # # --------
    # # COMMENT: Not used
    # # --------
    # def profitability_market(self, old_profits, profits,
    #                          lower_bound=-0.15, upper_bound=0.15):
    #     """TODO: write description.

    #     Args:
    #         old_profits     :
    #         profits         :
    #         lower_bound     :
    #         upper_bound     :

    #     Returns:
    #         profitability   :
    #     """
    #     profitability = 0
    #     if profits > 0 and old_profits > 0 and profits > old_profits:
    #         profitability = min(upper_bound,
    #                             np.log(1 + profits) -
    #                             np.log(1 + old_profits))
    #     return profitability

    # # --------
    # # COMMENT: Not used?
    # # --------
    # def number_entry(self, profitability, number_incumbents, o=0.5):
    #     return (number_incumbents * ((1 - o) * profitability +
    #             o * uniform.rvs(-0.15, 0.15)))

    # ------------------------------------------------------------------------
    #          VARIABLES CALCULATED AT CENTRAL LEVEL TO IMPROVE SPEED
    # ------------------------------------------------------------------------
    def wage_and_cons_unempl_pop(self):
        """Calculate wages, consumption and employment.

            COMMENT: look into more comprehensive structure later.
                     TODO: add more comments

                     NOTE that this function does compute values for
                     region 1, now sort of representing "other" region?
                     --> Can be changed to work more flexible
            TODO: check efficiency of this function
        """

        # -- WAGES -- #
        salaries = self.average_wage_regions(self.model.firms_1_2, True)
        self.salaries_cons = self.average_wage_regions(self.model.firms2)
        self.salaries_cap = self.average_wage_regions(self.model.firms1)
        self.salaries_serv = self.average_wage_regions(self.model.firms3)
        # salary_difference = self.variable_difference(av_sal0, av_sal1, True)
        # salary_difference0, salary_difference1 = salary_difference
        self.average_wages = [salaries[0], salaries[1], 0, 0]
        # av_sal0 = (salaries[0] / (self.av_price_cons[0] * 0.35 +
        #            self.av_price_serv[0] * 0.65)
        # av_sal1 = (self.pre_average_wages[0] / (self.av_price_cons[0] * 0.35 +
        #            self.av_price_serv[0] * 0.65)

        # -- EMPLOYMENT -- #
        RAE0 = salaries[2]
        RAE1 = salaries[3]
        self.prev_aggregate_employment = self.aggregate_employment
        self.aggregate_employment = [RAE0, RAE1]

        households = self.model.schedule.agents_by_type["Household"]
        unemployment_vars = self.aggregate_unemployment_regions(households)
        ARU0 = unemployment_vars[0]
        ARU1 = unemployment_vars[1]
        test_0 = unemployment_vars[2]
        test_1 = unemployment_vars[3]
        CCA_total = unemployment_vars[4]
        self.aggregate_unemployment = [ARU0, ARU1]
        self.CCAs = [CCA_total, unemployment_vars[5], unemployment_vars[6],
                     unemployment_vars[7], unemployment_vars[8]]

        # -- POPULATION -- #
        # Households
        self.regional_pop_hous = [ARU0 + RAE0, ARU1 + RAE1]
        unemployment_rate_0 = round(max(1, ARU0) /
                                    max(1, self.regional_pop_hous[0]), 2)
        unemployment_rate_1 = round(max(1, ARU1) /
                                    max(1, self.regional_pop_hous[1]), 2)
        self.regional_pop_hous = [ARU0 + RAE0, ARU1 + RAE1]

        unemployment_rate_0 = round(max(1, ARU0) / max(1, self.regional_pop_hous[0]), 2)
        unemployment_rate_1 = round(max(1, ARU1) / max(1, self.regional_pop_hous[1]), 2)
        unemployment_rate_total = (ARU0 + ARU1) / sum(self.regional_pop_hous)

        # Determine unemployment rates of form
        # [rate coastal, rate inland, unemployment diff coastal,
        #  unemployent diff inland, rate total]
        self.prev_unemployment_rates = self.unemployment_rates
        self.unemployment_rates = [round(unemployment_rate_0, 2),
                                   round(unemployment_rate_1, 2),
                                   0, 0, unemployment_rate_total]
        self.unemployment_cost = [ARU0 * self.unempl_subsidy[0],
                                  ARU1 * self.unempl_subsidy[1]]
        # unemployment_diff = self.variable_difference(unemployment_rate_0,
        #                                              unemployment_rate_1,
        #                                              False)
        # unemployment_diff0, unemployment_diff1 = unemployment_diff

        # -- CONSUMPTION -- #
        C0 = test_0
        # C0 = ((self.average_wages[0] * RAE0) +
        #       (ARU0 * self.unempl_subsidy[0]))
        C1 = test_1
        # C1 = ((self.average_wages[1] * RAE1) +
        #       (ARU1 * self.unempl_subsidy[1]))
        # test_0 = test_0 - C0
        # test_1 = test_1 - C1

        entry_exit_resources_c = (self.model.bailout_cost_coastal -
                                  self.model.new_firms_resources_coastal)
        # entry_exit_resources_i = (self.model.bailout_cost_inland -
        #                           self.model.new_firms_resources_inland)
        if C0 < entry_exit_resources_c:  # or C1 < entry_exit_resources_i:
            print("More resources than consumption")
        C0 += entry_exit_resources_c
        # C1 += entry_exit_resources_i

        # # Handle flood
        # if self.model.S > 0:
        #     if self.flood== True:
        #         shock = self.model.S
        #         # shock = np.random.beta(self.model.beta_a,
        #         #                        self.model.beta_b)
        #         C0 = (1 - shock) * C0

        fraction_cons_in_goods = 0.3
        G0 = fraction_cons_in_goods * C0
        G1 = fraction_cons_in_goods * C1
        S0 = C0 - G0
        S1 = C1 - G1

        # -- EXPORT -- #
        self.export_demand = self.export_demand * (1 + self.fraction_exp)
        # self.export_demand = (C0 + C1)  * self.fraction_exp
        export_demand_cons = self.export_demand * fraction_cons_in_goods
        export_demand_serv = self.export_demand - export_demand_cons
        # self.export_demand_list.append(self.export_demand)
        data = self.model.datacollector.model_vars
        region_market_share = data["Regional_sum_market_share"]
        exp_share = region_market_share[int(self.model.schedule.time)]
        # C0 += round(export_demand  * exp_share[3], 3)
        # C1 += round(export_demand *  exp_share[4], 3)

        self.aggregate_cons = [round(G0, 3), round(G1, 3),
                               G0 + G1, export_demand_cons]
        self.aggregate_serv = [round(S0, 3), round(S1, 3),
                               S0 + S1, export_demand_serv]
        self.test_cons = [test_0, test_1]

        # -- INCOME -- #
        self.previous_income_pp = self.income_pp
        self.income_pp = ((G0 / self.av_price_cons[0] +
                           S0 / self.av_price_serv[0]) /
                          self.regional_pop_hous[0])

        inc = self.variable_difference(self.income_pp,
                                       self.previous_income_pp)
        ld = self.variable_difference(1 - self.unemployment_rates[0],
                                      1 - self.prev_unemployment_rates[0])
        inc_pos, inc_neg = inc
        ld_pos, ld_neg = ld

        # unem_ratio = (self.unemployment_rates[0] /
        #               (self.prev_unemployment_rates[1] + 1e-5))

        # COMMENT: these if-statements were rewritten in last version
        mp = mn = 0
        # if (inc_pos < 0 and ld_pos < 0 and
        #         self.prev_unemployment_rates[0] > self.unemployment_rates[0]):
        if inc_pos < 0 and ld_pos < 0:
            if self.unemployment_rates[0] >= 0.2:
                mp = 0
            else:
                mp = 1 - np.exp(0.999 * inc_pos + 0.001 * ld_pos)
        # if (inc_neg < 0 and ld_neg < 0 and
        #         self.prev_unemployment_rates[0] > self.unemployment_rates[0]):
        if inc_neg < 0 and ld_neg < 0:
            if self.unemployment_rates[0] <= 0.05:
                mn = 0
            else:
                mn = 1 - np.exp(0.999 * inc_neg + 0.001 * ld_neg)

        if self.previous_income_pp > 0:
            self.income_pp_change = ((self.income_pp -
                                      self.previous_income_pp) /
                                     self.previous_income_pp)
        else:
            self.income_pp_change = 0

    def variable_difference(self, var0, var1, prefer_foreign_vars=True):
        """TODO: write description.

        Args:
            var0                    :
            var1                    :
            prefer_foreign_vars     :
        """
        # --------
        # COMMENT: Only returns nonzero value if prefer_foreign_vars is True?
        #          --> Change to only use it in this case?
        # TODO: check efficiency of this function
        # --------

        var_diff0 = var_diff1 = 0
        if prefer_foreign_vars:
            if var1 > var0:
                var_diff0 = (var0 - var1) / (var0 + 0.001)
            elif var0 > var1:
                var_diff1 = (var1 - var0) / (var1 + 0.001)
        # else:
        #     if var1 < var0:
        #         var_diff0 = (var1 - var0) / (var1 + 0.001)
        #     elif var0 < var1:
        #         var_diff1 = (var0 - var1) / (var1 + 0.001)

        # var_diff = abs(var0 - var1)
        # if var_diff > 0.1:
        #     var_diff_perc = var_diff / (var0 + var1) / 2
        # elif var_diff > 0.01:
        #     var_diff0 = round((var0 - var1) / max(var0, var1), 2)
        #     var_diff1 = round((var1 - var0) / max(var0, var1), 2)
        # else:
        #     var_diff_perc = 0
        return (round(max(-0.25, var_diff0), 2),
                round(max(-0.25, var_diff1), 2))

    def compute_productivity(self):
        """TODO: write description.
           TODO: look into efficiency of this function.
           TODO: divide into separate functions?
        """

        # -------
        # COMMENT: look into faster way to do this (python map function?)
        # -------
        # LD0 = LD1 = 0
        LD0_cap = LD1_cap = 0
        LD0_serv = LD1_serv = 0
        LD0_cons = LD1_cons = 0
        AE_cap_0 = AE_cap_1 = 0
        AE_cons_0 = AE_cons_1 = 0
        AE_serv_0 = AE_serv_1 = 0
        region0_cap = region1_cap = 0
        region0_cons = region1_cons = 0
        region0_serv = region1_serv = 0
        total_prod_cap_0 = total_prod_cap_1 = 0
        total_prod_cons_0 = total_prod_cons_1 = 0
        total_prod_serv_0 = total_prod_serv_1 = 0
        feas_prod_cons_0 = feas_prod_cons_1 = 0
        feas_prod_serv_0 = feas_prod_serv_1 = 0
        des_prod_serv_0 = des_prod_serv_1 = 0
        real_demand_cap = real_demand_serv = 0
        real_demand_cons = quantity_ordered = 0
        profits_cons_0 = profits_cons_1 = 0
        profits_cap_0 = profits_cap_1 = 0
        profits_serv_0 = profits_serv_1 = 0

        GDP_cons_0 = GDP_cons_1 = 0
        GDP_cap_0 = GDP_cap_1 = 0
        GDP_serv_0 = GDP_serv_1 = 0

        num_offer_cons_0 = num_offer_cons_1 = 0
        num_offer_serv_0 = num_offer_serv_1 = 0

        NW1 = 0
        NW0 = 0

        for firm in self.model.firms_1_2:
            if firm.region == 0:
                # LD0 += firm.labor_demand
                NW0 += firm.net_worth
                if firm.type == "Cap":
                    LD0_cap += firm.labor_demand
                    region0_cap += 1
                    AE_cap_0 += len(firm.employees_IDs)
                    total_prod_cap_0 += firm.production_made
                    real_demand_cap += sum(firm.real_demand_cap)
                    profits_cap_0 += firm.profits
                elif firm.type == "Service":
                    LD0_serv += firm.labor_demand
                    region0_serv += 1
                    AE_serv_0 += len(firm.employees_IDs)
                    total_prod_serv_0 += firm.production_made
                    real_demand_serv += firm.real_demand
                    feas_prod_serv_0 += firm.feasible_production
                    profits_serv_0 += firm.profits
                    num_offer_serv_0 += len(firm.offers)
                elif firm.type == "Cons":
                    LD0_cons += firm.labor_demand
                    region0_cons += 1
                    AE_cons_0 += len(firm.employees_IDs)
                    total_prod_cons_0 += firm.production_made
                    feas_prod_cons_0 += firm.feasible_production
                    real_demand_cons += firm.real_demand
                    quantity_ordered += firm.quantity_ordered
                    profits_cons_0 += firm.profits
                    num_offer_cons_0 += len(firm.offers)

            elif firm.region == 1:
                # LD1 += firm.labor_demand
                NW1 += firm.net_worth
                if firm.type == "Cap":
                    LD1_cap += firm.labor_demand
                    region1_cap += 1
                    AE_cap_1 += len(firm.employees_IDs)
                    total_prod_cap_1 += firm.production_made
                    real_demand_cap += sum(firm.real_demand_cap)
                    profits_cap_1 += firm.profits
                elif firm.type == "Service":
                    LD1_serv += firm.labor_demand
                    region1_serv += 1
                    AE_serv_1 += len(firm.employees_IDs)
                    total_prod_serv_1 += firm.production_made
                    real_demand_serv += firm.real_demand
                    feas_prod_serv_1 += firm.feasible_production
                    # des_prod_serv_1 += firm.desired_production
                    profits_serv_1 += firm.profits
                    num_offer_serv_1 += len(firm.offers)
                elif firm.type == "Cons":
                    LD1_cons += firm.labor_demand
                    region1_cons += 1
                    AE_cons_1 += len(firm.employees_IDs)
                    total_prod_cons_1 += firm.production_made
                    feas_prod_cons_1 += firm.feasible_production
                    real_demand_cons += firm.real_demand
                    quantity_ordered += firm.quantity_ordered
                    profits_cons_1 += firm.profits
                    num_offer_cons_1 += len(firm.offers)

        # TODO: comment
        self.total_offers = [num_offer_cons_0, num_offer_cons_1,
                             num_offer_serv_0, num_offer_serv_1]
        productivity0_old = max(1, self.regional_av_prod[0])
        productivity1_old = max(1, self.regional_av_prod[1])
        self.old_profits_firms = self.profits_firms
        self.profits_firms = [profits_cap_0, profits_cap_1,
                              profits_cons_0, profits_cons_1,
                              profits_serv_0, profits_serv_1]
        self.orders = quantity_ordered
        self.real_demands = [real_demand_cap,
                             real_demand_cons,
                             real_demand_serv]
        AE_0 = AE_cons_0 + AE_cap_0 + AE_serv_0
        AE_1 = AE_cons_1 + AE_cap_1 + AE_serv_1
        self.total_productions = [total_prod_cap_0, total_prod_cap_1,
                                  total_prod_cons_0, total_prod_cons_1,
                                  total_prod_serv_0, total_prod_serv_1]

        total_prod_0 = total_prod_cons_0 + total_prod_cap_0 + total_prod_serv_0
        total_prod_1 = total_prod_cons_1 + total_prod_cap_1 + total_prod_serv_1

        def prod_increase(aggr_empl, total_prod, prod_old,
                          lower_bound=-0.25):
            """Calculate increase in production.

            Args:
                aggr_empl           : Total number of employess for all firms
                total_prod          : Total production for all firms
                prod_old            : Production in previous timestep
                lower_bound         : Production increase lower bound

            Returns:
                av_prod             : Average productivity per region
                production_increase : Increase in average production compared
                                      to previous timestep
            """
            upper_bound = -1.25 * lower_bound
            if aggr_empl > 0:
                av_prod = total_prod / aggr_empl
                production_increase = max(lower_bound,
                                          min(upper_bound,
                                              (av_prod - prod_old) /
                                              prod_old))
            else:
                av_prod = 0
                production_increase = 0
            return av_prod, production_increase

        av_prod0, prod_increase_0 = prod_increase(AE_0, total_prod_0,
                                                  productivity0_old)
        av_prod1, prod_increase_1 = prod_increase(AE_1, total_prod_1,
                                                  productivity1_old)

        self.regional_av_prod = [av_prod0, av_prod1,
                                 prod_increase_0, prod_increase_1]

        # Calculate average production, consumption good firms
        productivity_cons_old = self.regional_av_prod_cons
        productivity_cons_0_old = max(1, productivity_cons_old[0])
        productivity_cons_1_old = max(1, productivity_cons_old[1])

        prod_increase_vars = prod_increase(AE_cons_0, total_prod_cons_0,
                                           productivity_cons_0_old)
        av_prod_cons_0, prod_increase_cons_0 = prod_increase_vars
        prod_increase_vars = prod_increase(AE_cons_1, total_prod_cons_1,
                                           productivity_cons_1_old)
        av_prod_cons_1, prod_increase_cons_1 = prod_increase_vars
        self.regional_av_prod_cons = [av_prod_cons_0,
                                      av_prod_cons_1,
                                      prod_increase_cons_0,
                                      prod_increase_cons_1]

        # Calculate average production, capital good firms
        productivity_cap_old = self.regional_av_prod_cap
        productivity_cap_0_old = max(1, productivity_cap_old[0])
        productivity_cap_1_old = max(1, productivity_cap_old[1])

        prod_increase_vars = prod_increase(AE_cap_0, total_prod_cap_0,
                                           productivity_cap_0_old)
        av_prod_cap_0, prod_increase_cap_0 = prod_increase_vars
        prod_increase_vars = prod_increase(AE_cap_1, total_prod_cap_1,
                                           productivity_cap_1_old)
        av_prod_cap_1, prod_increase_cap_1 = prod_increase_vars
        self.regional_av_prod_cap = [av_prod_cap_0, av_prod_cap_1,
                                     prod_increase_cap_0, prod_increase_cap_1]

        # Calculate average production, service firms
        productivity_serv_old = self.regional_av_prod_serv
        productivity_serv_0_old = max(1, productivity_serv_old[0])
        productivity_serv_1_old = max(1, productivity_serv_old[1])

        prod_increase_vars = prod_increase(AE_serv_0, total_prod_serv_0,
                                           productivity_serv_0_old)
        av_prod_serv_0, prod_increase_serv_0 = prod_increase_vars
        prod_increase_vars = prod_increase(AE_serv_1, total_prod_serv_1,
                                           productivity_serv_1_old)
        av_prod_serv_1, prod_increase_serv_1 = prod_increase_vars
        self.regional_av_prod_serv = [av_prod_serv_0,
                                      av_prod_serv_1,
                                      prod_increase_serv_0,
                                      prod_increase_serv_1]

        # Calculate feasible and desired productions
        self.feasible_productions = [feas_prod_cons_0, feas_prod_cons_1,
                                     feas_prod_serv_0, feas_prod_serv_1]
        self.desired_productions = [0, 0, des_prod_serv_0, des_prod_serv_1]

        LD0 = round(LD0_cap + LD0_serv + LD0_cons)
        LD1 = round(LD1_cap + LD1_serv + LD1_cons)
        ld_diffrence0, ld_diffrence1 = self.variable_difference(LD0, LD1, True)
        self.labor_demand_difference = [ld_diffrence0, ld_diffrence1]
        self.labour_demands = [LD0, LD1,
                               LD0_cap, LD1_cap,
                               LD0_serv, LD1_serv,
                               LD0_cons, LD1_cons]
        self.firms_pop = [region0_cap, region1_cap,
                          region0_cons, region1_cons,
                          region0_serv, region1_serv]
        self.av_size = [self.aggregate_employment[0] /
                        max(1, (self.firms_pop[0] +
                                self.firms_pop[2] +
                                self.firms_pop[4])),
                        self.aggregate_employment[1] /
                        max(1, (self.firms_pop[1] +
                                self.firms_pop[3] +
                                self.firms_pop[5]))]
        self.av_net_worth = [NW0 /
                             (self.firms_pop[0] + self.firms_pop[2] +
                              self.firms_pop[4] + 1),
                             NW1 / (self.firms_pop[1] + self.firms_pop[3] +
                                    self.firms_pop[5] + 1)]
        # self.cap_av_prod = self.productivity_firms(self.model.firms1)
        # NW0 = 0

    def compute_prices(self):
        """Calculates average prices for all firm types. """
        self.av_price_cap = self.weighted_price_avg(self.model.firms1)
        self.av_price_cons = self.weighted_price_avg(self.model.firms2)
        self.av_price_serv = self.weighted_price_avg(self.model.firms3)
        # --------
        # COMMENT: not for service firms?
        # --------
        self.av_price = self.weighted_price_avg(self.model.firms_1_2)

    def compute_net_sales(self, firms):
        """Calculates net sales for ConsumptionGood firms.

        Args:
            firms       : List of Firm objects
        """

        # Get sales of consumption good firms per region
        sales = np.array([sum(firm.sales - firm.total_costs for firm in firms
                              if firm.region == 0),
                          sum(firm.sales - firm.total_costs for firm in firms
                              if firm.region == 1)])
        # Convert to log values
        self.net_sales_cons_firms = np.around(np.where(sales > 0,
                                                       np.log(sales),
                                                       0), 3)

    # -----------------------
    #     HELPER FUNCTIONS
    # -----------------------
    def weighted_price_avg(self, firms):
        """Calculates average price of given firms, weighted by market share. """
        price0 = 0
        price1 = 0
        for firm in firms:
            price0 += firm.price * firm.market_share[0]
            price1 += firm.price * firm.market_share[1]
        return [abs(round(price0, 5)), abs(round(price1, 5))]

    def average_wage_regions(self, agents_class, aggregate=False):
        """TODO: write description. """
        var_0 = den_0 = av_var_0 = 0
        var_1 = den_1 = av_var_1 = 0
        for i in range(len(agents_class)):
            agent = agents_class[i]
            if agent.region == 0:
                var_0 += agent.wage * len(agent.employees_IDs)
                den_0 += len(agent.employees_IDs)

            if agent.region == 1:
                var_1 += agent.wage * len(agent.employees_IDs)
                den_1 += len(agent.employees_IDs)

        if den_0 != 0:
            av_var_0 = round(var_0 / den_0, 2)

        if den_1 != 0:
            av_var_1 = round(var_1 / den_1, 2)

        if not aggregate:
            return [av_var_0, av_var_1]
        else:
            return [av_var_0, av_var_1, den_0, den_1]

    def aggregate_unemployment_regions(self, households):
        """TODO: write description.

        Args:
            households  : Dict containing all households
        """
        ARE0 = ARE1 = 0
        AC0 = AC1 = 0
        CCA = 0
        elevation = des_elevation = 0
        wet_proof = 0
        dry_proof = 0
        for h in households.values():
            if h.region == 0:
                CCA += h.CCA
                AC0 += h.consumption
                if h.employer_ID is None:
                    ARE0 += 1
                elevation += h.elevated
                dry_proof += h.dry_proofed
                wet_proof += h.wet_proofed
            elif h.region == 1:
                AC1 += h.consumption
                if h.employer_ID is None:
                    ARE1 += 1
        return (ARE0, ARE1, AC0, AC1, CCA, elevation,
                des_elevation, dry_proof, wet_proof)

    # ------------------------------------------------------------------------
    # Some functions for future flows of climate hazards (not singular events)
    # ------------------------------------------------------------------------
    # def climate_damages(self):
    #     self.flooded = False
    #     time_current = int(self.model.schedule.time)
    #     if (self.region == 0 and
    #             (time_current == self.model.shock_time or
    #              time_current == self.model.shock_time + 100)):
    #         self.flooded = True
    #         print("gov flood")

    # def climate_damages(self):
    #     # self.flooded = False
    #     time_current = int(self.model.schedule.time)
    #     if self.region == 0 and time_current == self.model.shock_time:
    #         firms = self.model.firms_1_2
    #         print("Flooding all the firms")
    #         for i in range(len(firms)):
    #             if firms[i].region == 0:
    #                 firms[i].flooded = True

    #     if self.region == 0 and time_current == self.model.shock_time + 1:
    #         firms = self.model.firms_1_2
    #         print("Remove flooding from all the firms")
    #         for i in range(len(firms)):
    #             if firms[i].flooded == True:
    #                 firms[i].flooded = False

    # ------------------------------------------------------------------------
    #                   GOVERNMENT STAGES FOR STAGED ACTIVATION
    # ------------------------------------------------------------------------
    def stage0(self):
        if self.region == 0:
            self.minimum_wage()
            self.fiscal_balance = (self.tax_revenues[0] -
                                   self.unemployment_cost[0])

            # marginal = 0
            # if self.model.time > 70:
            #     marginal = 0.01
            # if self.fiscal_balance > 0:
            #     self.tax_rate = (1 - marginal) * self.tax_rate
            # elif self.fiscal_balance < 0:
            #     self.tax_rate = (1 + marginal) * self.tax_rate

            self.tax_revenues = [0, 0]
            self.unemployment_cost = [0, 0]

            # agents = self.model.firms2
            # for i in range(len(agents)):
            #     agent = agents[i]
            #     agent.market_share[2] = 1/len(agents)

    def stage1(self):
        self.entrants_cons_coastal = 0
        self.entrants_cons_inland = 0
        self.entrants_serv_coastal = 0
        self.entrants_serv_inland = 0
        self.open_vacancies_list()

    def stage2(self):
        if self.region == 0:
            temp = self.norm_price_unf_demand(self.model.firms2,
                                              self.model.ids_firms2)
            self.norm_price_unfilled_demand = temp
            temp = self.norm_price_unf_demand(self.model.firms3,
                                              self.model.ids_firms3)
            self.norm_price_unfilled_demand_serv = temp

    def stage3(self):
        if self.region == 0:
            comp_norm = self.norm_competitiveness(self.model.firms2)
            self.comp_normalized = comp_norm[0]
            self.average_normalized_comp = comp_norm[1]
            comp_norm = self.norm_competitiveness(self.model.firms3)
            self.serv_comp_normalized = comp_norm[0]
            self.serv_average_normalized_comp = comp_norm[1]
            self.norm_market_share_cap(self.model.firms1)
            self.compute_prices()
            self.wage_and_cons_unempl_pop()
            # self.compute_productivity()

    def stage4(self):
        if self.region == 0:
            MS_norm = self.norm_market_share(self.model.firms2)
            self.average_normalized_market_share = MS_norm[0]
            self.market_shares_normalized = MS_norm[1]
            MS_norm = self.norm_market_share(self.model.firms3)
            self.average_normalized_market_share_serv = MS_norm[0]
            self.market_shares_normalized_serv = MS_norm[1]

    def stage5(self):
        if self.region == 0:
            self.compute_net_sales(self.model.firms2)
            # self.entry_exit_cons()
            # MS_norm = self.norm_market_share(self.model.firms2)
            # self.average_normalized_market_share = MS_norm
            # self.market_shares_normalized = MS_norm
            # MS_norm = self.norm_market_share(self.model.firms3)
            # self.average_normalized_market_share_serv = MS_norm
            # self.market_shares_normalized_serv = MS_norm
            self.get_best_cap()
            # self.norm_market_share_cap(self.model.firms1)
            self.compute_productivity()

    def stage6(self):
        pass