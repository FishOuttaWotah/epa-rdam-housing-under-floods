from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import bisect
import random
from random import sample, seed, uniform, shuffle
from math import exp, sqrt
from scipy.stats import bernoulli
from scipy.stats import beta

if TYPE_CHECKING:
    import model_init
from agent_base import CustomAgent
import labor_dynamics as ld
import goods_market as gm
import climate_dynamics as cd  # TODO: this will be depreciated
import accounting as ac
import vintage as vin
import research_and_development as rd

"""
__________________________________________

Firm Agent, basic
__________________________________________

"""
"""
This is a combination of the firm agent types (capital goods, consumption goods, service goods) from Alessandro's CRAB model into a generic firm agent.  
"""


class Firm(CustomAgent):
    firm_type = None

    def __init__(self, unique_id: str, location: str, model: model_init.RHuCC_Model, flood_exp: tuple[float],
                 flood_dmg: tuple[float],
                 productivity: float, wage: float, price: float, initial_net_worth: float):
        # basic Mesa-level-like attributes
        super(Firm, self).__init__(unique_id=unique_id, model=model)

        # common attributes
        self.location = location  # NEW
        # self.firm_type: str = firm_typeg
        self.house_quarter_income_ratio = None
        self.repair_exp = None
        self.consumption = None
        self.monetary_damage = None
        self.total_damage = None
        # self.flooded changed to 'self.is_flooded' below
        self.at_risk = None
        self.worry = None
        self.risk_perc = None
        self.SE_wet = None
        self.RE_wet = None
        self.total_savings_pre_flood = None
        self.productivity = [productivity, productivity]
        self.region = 0  # NB SL: added for dummy, may have consequences for calculation
        self.cost = 0
        self.wage = wage
        self.price = price
        self.sales = 0
        self.net_worth = initial_net_worth
        self.lifecycle = 1
        self.subsidiary = 0
        # self.height and self.damage_coeff moved to climate and flooding section
        self.net_worth = 0

        # LIFETIME AND MIGRATION (not used)
        # self.distances_mig = []
        # self.region_history = []
        # self.migration_pr = 0
        # self.pre_shock_prod = 0
        # self.new_born = False

        # LABOR MARKET, DERIVED FROM ALESSANDRO'S CRAB MODEL
        self.employees = []
        self.desired_employees = None  # will be filled by ld hire/fire method
        self.employees_IDs = None
        self.open_vacancies = False
        self.wages = []  # changed: this one is dependent on households' socioeconomic level
        self.labor_demand = 0.

        # CLIMATE AND FLOODING-BASED ATTRIBUTES
        self.height = 0  # NOT USED
        self.flood_exp = flood_exp  # NEW\
        self.flood_dmg = flood_dmg
        self.damage_coeff = 0  # WOULD BE MODIFIED WITH NEW MECHANISM
        self.is_flooded = 0
        self.flood_damage = []  # OVERLAP WITH NEW MECHANISM
        self.recovery_time = None
        self.renounced_cons = None
        self.CCA_resilience = [1, 1]
        self.CCA_RD_budget = 0

    def hire_or_fire(self):
        # this is a common method that must be filled/replaced, otherwise it'll raise a NotImplementedError
        raise NotImplementedError("Agent method for hire_or_fire is not provided in agent implementation")

        pass


class ConsumptionFirm(Firm):
    """

    """

    stage_list = []  # generic retrievable attribute for scheduling
    firm_type = None

    # might need to include a check (at instance generation time perhaps) to see if class contains the method(s)
    def __init__(self,
                 unique_id: str,
                 model: model_init.RHuCC_Model,
                 location: str,
                 productivity: float,
                 wage: float,
                 price: float,
                 initial_net_worth: float,
                 initial_capital_output_ratio: float,
                 initial_machines: int,
                 initial_amount_capital: float,  # dropped initial_number_of_machines, possibly redundant
                 flood_exp: tuple[float],
                 flood_dmg: tuple[float]):
        super().__init__(unique_id=unique_id, location=location, model=model, flood_exp=flood_exp, flood_dmg=flood_dmg,
                         productivity=productivity, wage=wage, price=price, initial_net_worth=initial_net_worth)

        self.debt = None
        self.capital_vintage = [vin.Vintage(productivity, initial_amount_capital) for i in
                                range(initial_machines)]

        # -- Production -- #
        # Please note that for the variables that differ for region
        # (i.e. demands), element [0] refers to the Coastal region
        # and element [1] refers to the Inland region
        self.past_demands = [1, 1]
        self.feasible_production = 0
        self.desired_production = None
        self.price = price
        self.normalized_price = 0
        self.markup = 0.14
        self.cost = 0
        self.total_costs = 0
        self.production_made = 1
        self.inventories = 0
        self.productivity = [productivity, productivity]
        self.order_canceled = False
        self.order_reduced = 0
        self.capital_output_ratio = initial_capital_output_ratio  # !!! move from model-level to explicit function call
        self.capital_amount = (initial_machines *
                               initial_amount_capital)  # changed to explicit function call

        # -- Investment -- #
        self.expansion = 0
        self.replacements = 0
        self.scrapping_machines = 0
        self.investment_cost = 0
        self.quantity_ordered = 0
        # Pick random supplier, get offers and assign self to supplier clients
        capital_firm_dict = self.model.schedule.agents_by_type[CapitalFirm.__name__]  # MODIFIED
        self.supplier_id = random.choice(list(capital_firm_dict))
        self.offers = [capital_firm_dict[self.supplier_id].brochure_regional]
        capital_firm_dict[self.supplier_id].client_IDs.append(self.unique_id)

        # -- Goods market -- #
        self.competitiveness = [1, 1]
        self.market_share = None  # TODO: update market_share for service firm
        self.market_share_history = []
        self.regional_demand = [0, 0, 0]
        self.past_sales = self.regional_demand
        self.demand = 0
        self.real_demand = 1
        self.sales = 10
        self.filled_demand = 0
        self.unfilled_demand = 0
        self.profits = 0
        self.unfilled_calib = 0
        self.number_out = 0

    # clean up inputs and document changes wrt. Alessandro's CRAB model
    def hire_or_fire(self):

        # specific for consumption firms. The assumption is that these firms have the same generic firm attributes
        """Open vacancies and fire employees based on demand.

                """
        # Keep track of old productivity
        self.productivity[0] = self.productivity[1]

        # Determine labor demand and hire employees
        if self.feasible_production > 0:
            labor_info = ld.labor_demand(self.capital_vintage,
                                         self.feasible_production)
            self.labor_demand, self.productivity[1] = labor_info
        else:
            self.labor_demand = 0
            self.productivity[1] = self.productivity[0]
        # # If we want to add the coast of research once we add the CCA
        # self.labor_demand += math.floor(self.CCA_RD_budget / self.wage)

        employee_info = ld.hire_and_fire(self.labor_demand,
                                         self.employees_IDs,
                                         self.open_vacancies,
                                         self.model, self.profits, self.wage)
        self.desired_employees = employee_info[0]
        self.employees_IDs = employee_info[1]
        self.open_vacancies = employee_info[2]

    def market_share_calculation(self):
        """Compute firm market share.
           COMMENT: now stored in government, here only retrieved from there.
        """
        # Retrieve competitiveness from central calculations of government
        gov = self.model.governments[0]
        # --------
        # COMMENT: why noise addition?
        # --------
        self.competitiveness = [comp + 1e-7 for comp in
                                gov.comp_normalized[self.unique_id]]
        avg_comp = gov.average_normalized_comp
        # self.capital_amount = sum(i.amount for i in self.capital_vintage)
        # # OR:
        # if len(self.employees_IDs) > 0 or self.lifecycle < 10:
        #     self.capital_amount = sum(i.amount for i in self.capital_vintage)

        # Compute market share from competitiveness
        K_total = False
        a = 1
        # Make the market more stable at the beginning, for a smooth start
        if self.model.schedule.steps < 50:
            a = 0.5
        if self.lifecycle == 0:
            data = self.model.datacollector.model_vars["Capital_Region_cons"]
            K_total = data[int(self.model.schedule.time)]

        capital_stock = self.capital_amount / self.capital_output_ratio
        self.market_share = gm.calc_market_share_cons(self.model,
                                                      self.lifecycle,
                                                      self.market_share,
                                                      self.competitiveness,
                                                      avg_comp,
                                                      capital_stock,
                                                      K_total, a)

    def price_demand_normalized(self):
        """Retrieves normalized price and unfilled demand
           from the government, that does it at central level
        """
        gov = self.model.governments[0]
        price = gov.norm_price_unfilled_demand[self.unique_id]
        self.normalized_price = round(price[0], 8)
        self.unfilled_demand = round(price[1], 8)

    def market_share_normalized(self):
        """Retrieves normalized market shares from the government. """
        gov = self.model.governments[0]
        market_share = gov.market_shares_normalized[self.unique_id]
        self.market_share = [round(i, 8) for i in market_share]

    def market_share_trend(self):
        """TODO: write description
           (Or remove function, it is only one line.
            Or add to function above?).
        """
        self.market_share_history.append(round(sum(self.market_share), 5))

    def accounting(self):
        """Calculates individual demand, compares to
           production made and accounting costs, sales and profits.
        """
        # -- Accounting -- #
        # NOTE: CCA budget = 0 now
        self.total_costs = round(self.production_made * self.cost, 3)
        self.sales = round(self.demand_filled * self.price, 3)

        # Cancel orders that cannot be fulfilled by supplier
        if self.order_canceled:
            self.scrapping_machines = 0
            if self.debt > 0:
                self.debt = max(0, self.debt - self.investment_cost)
            self.investment_cost = 0
            self.quantity_ordered = 0
            self.order_canceled = False

        # Reduce orders that cannot be fulfilled by supplier
        if self.order_reduced > 0 and self.supplier_id is not None:
            supplier = self.model.schedule.agents[self.supplier_id]
            self.quantity_ordered = max(0, self.quantity_ordered -
                                        self.order_reduced)
            self.scrapping_machines -= self.order_reduced
            self.investment_cost = self.quantity_ordered * supplier.price
            if self.debt > 0:
                self.debt = max(0, self.debt -
                                self.order_reduced * supplier.price)
            self.order_reduced = 0

        # Compute profits
        self.profits = round(self.sales - self.total_costs -
                             self.debt * (1 + self.model.interest_rate), 3)

        # If profits are positive: pay taxes
        if self.profits > 0:
            gov = self.model.governments[0]
            tax = gov.tax_rate * self.profits
            self.profits = self.profits - tax
            gov.tax_revenues[self.region] += tax

        # Add earnings to net worth
        self.net_worth += self.profits - self.investment_cost
        # If new worth is positive firm is not credit constrained
        if self.net_worth > 0:
            self.credit_rationed = False

    def stage0(self):
        """Stage 0:
           TODO: write short description for all stages
        """

        # There is a minimum period before considering migration
        # --------
        # COMMENT: 1. Migration now does not happen for ServiceFirms?
        #          2. Lifecycle stuff is same for firms --> move to
        #             other function?
        # --------
        self.migration = False

        # if self.lifecycle > 16:  # or self.new_born == True:
        #     migr_prob = migration.migration_probability(self.regional_demand,
        #                                                 self.past_sales,
        #                                                 self.region)

        #     if migr_prob > 0:
        #         new_attr = migration.firm_migrate(self,
        #                                           migr_prob,
        #                                           self.model,
        #                                           self.region,
        #                                           self.unique_id,
        #                                           self.employees_IDs,
        #                                           self.net_worth,
        #                                           self.wage)
        #         self.region = new_attr[0]
        #         self.employees_IDs = new_attr[1]
        #         self.net_worth = new_attr[2]
        #         self.wage = new_attr[3]

        # # If there are no workers in the region, the firm migrates
        # workers = self.model.governments[0].aggregate_employment[self.region]
        # if self.lifecycle > 4 and workers == 0:
        #     new_attr = migration.firm_migrate(self, 1,
        #                                       self.model,
        #                                       self.region,
        #                                       self.unique_id,
        #                                       self.employees_IDs,
        #                                       self.net_worth,
        #                                       self.wage)
        #     self.region = new_attr[0]
        #     self.employees_IDs = new_attr[1]
        #     self.net_worth = new_attr[2]
        #     self.wage = new_attr[3]

        # Handle flood if it occurs
        # --------
        # COMMENT: why set to 0 here?
        # --------
        self.damage_coeff = 0
        if self.model.is_flood_now:
            self.damage_coeff = cd.depth_to_damage(self.model.flood_depth,
                                                   self.height, self.type)
        if self.lifecycle > 0:
            vin.update_capital(self)
            self.quantity_ordered = 0
            self.debt = 0
            # Sum the quantity of each machine
            self.capital_amount = np.ceil(sum(i.amount for i in
                                              self.capital_vintage))
            new_attr = vin.capital_investments(self.model.time,
                                               self.capital_amount,
                                               self.capital_output_ratio,
                                               self.past_demands,
                                               self.real_demand,
                                               self.inventories,
                                               self.unfilled_demand)
            self.past_demands = new_attr[0]
            self.inventories = new_attr[1]
            self.unfilled_demand = new_attr[2]
            self.desired_production = new_attr[3]
            self.feasible_production = new_attr[4]
            self.expansion = new_attr[5]

            if self.damage_coeff > 0:
                self.inventories = cd.destroy_fraction(self.inventories,
                                                       self.damage_coeff)

            # Calculate replacement investment (function below)
            self.replacements = round(vin.calc_replacement_investment(self))
            if self.net_worth > (len(self.employees_IDs) * self.wage):
                if self.replacements > 0 or self.expansion > 0:
                    vin.choose_supplier_and_place_order(self)

    def stage1(self):
        """Stage 1:
           TODO: write short description for all stages
        """
        # # Not for newborn firms
        # if self.lifecycle > 0:
        gov = self.model.governments[0]
        self.wage = ld.wage_determination(self.region, gov,
                                          self.productivity,
                                          gov.regional_av_prod_serv,
                                          self.wage, self.price,
                                          self.lifecycle)
        self.hire_or_fire()  # LS: renamed

    def stage2(self):
        """Stage 2:
           TODO: write short description for all stages
        """
        if self.damage_coeff > 0:
            self.productivity[1] = cd.destroy_fraction(self.productivity[1],
                                                       self.damage_coeff)
        new_attr = gm.compete_and_sell(self.productivity,
                                       self.wage,
                                       self.market_share_history,
                                       self.price,
                                       self.markup)
        self.cost, self.markup, self.price = new_attr

    def stage3(self):
        """Stage 3:
           TODO: write short description for all stages
        """
        # if s.lifecycle > 0:
        households = self.model.schedule.agents_by_type["Household"]
        ld.update_employees_wage(self.employees_IDs, households, self.wage)
        self.price_demand_normalized()
        trade_cost = self.model.governments[0].transport_cost
        trade_cost_exp = self.model.governments[0].transport_cost_RoW

        self.competitiveness = gm.calc_competitiveness(self.normalized_price,
                                                       self.region,
                                                       trade_cost,
                                                       trade_cost_exp,
                                                       self.unfilled_demand)

    def stage4(self):
        """Stage 4:
           TODO: write short description for all stages
        """
        # if self.lifecycle > 0:
        self.market_share_calculation()

    def stage5(self):
        """Stage 5:
           TODO: write short description for all stages
        """
        # TODO: comment
        # if self.lifecycle > 0:
        self.market_share_normalized()
        self.market_share_trend()

        # TODO: comment
        # if len(self.employees_IDs) > 0 or self.lifecycle < 10:
        total_demand = self.model.governments[0].aggregate_serv
        new_attr = ac.individual_demands(len(self.employees_IDs),
                                         self.lifecycle,
                                         self.regional_demand,
                                         self.market_share,
                                         total_demand,
                                         self.price,
                                         self.productivity[1])
        self.monetary_demand = new_attr[0]
        self.regional_demand = new_attr[1]
        self.real_demand = new_attr[2]
        self.production_made = new_attr[3]
        self.past_sales = new_attr[4]

        # TODO: comment
        new_attr = ac.production_filled_unfilled(self.production_made,
                                                 self.inventories,
                                                 self.real_demand,
                                                 self.lifecycle)
        self.demand_filled, self.unfilled_demand, self.inventories = new_attr
        # if self.damage_coeff > 0:
        #     self.production_made = cd.destroy_fraction(self.production_made,
        #                                                self.damage_coeff)
        self.accounting()

    def stage6(self):
        """Stage 6:
           TODO: write short description for all stages
        """
        # # COMMENT: can be removed for all firms?
        # if self.lifecycle > 7:
        #     self.new_born = False

        # --------
        # COMMENT: should be 1, but same function as capital goods
        #          and there it is 2?
        # --------
        if self.lifecycle > 1:
            if len(self.employees_IDs) > 1:  # and self.model.time > 50:
                self.subsidiary = ac.new_entry(self.profits,
                                               len(self.employees_IDs),
                                               self.subsidiary, self.wage)
                if self.subsidiary > 3:  # and uniform(0, 1) > 0.5:
                    if self.region == 0:
                        # self.model.governments[0].entrants_serv_coastal +=1
                        self.model.subs_serv_coastal.append(self)
                    elif self.region == 1:
                        # self.model.governments[0].entrants_serv_inland +=1
                        self.model.subs_serv_inland.append(self)
                    self.subsidiary = 0
            # if (self.market_share[self.region] < 1e-6 or
            #         sum(self.past_demands) < 1 or
            #         (len(self.capital_vintage) == 0 and
            #          self.net_worth < 10)):
            # # OR:

            if (self.market_share[self.region] < 1e-6 or
                    sum(self.past_demands) < 1):
                if self.region == 0 and len(self.model.ids_firms3) > 10:
                    self.model.kill_serv_coastal.append(self)
                # else:
                #     self.model.kill_serv_inland.append(self)

                # Fire employees
                self.employees_IDs = ac.remove_employees(self.employees_IDs,
                                                         self.model.schedule)
                # COMMENT: TODO: change offer list to dict
                supplier_ids = [i[2] for i in self.offers]
                ac.remove_offers(supplier_ids, self.model.schedule,
                                 self.unique_id)

        # Update offer list and lifecycle
        self.offers = []
        self.lifecycle += 1


class CapitalFirm(Firm):
    firm_type = 'cap'

    def __init__(self,
                 unique_id: str,
                 model: model_init.RHuCC_Model,
                 location: str,
                 productivity: float,
                 wage: float,
                 price: float,
                 initial_net_worth: float,
                 transport_cost: float,
                 flood_exp: tuple[float],
                 flood_dmg: tuple[float]):
        super().__init__(unique_id=unique_id, location=location, model=model, flood_exp=flood_exp, flood_dmg=flood_dmg,
                         productivity=productivity, wage=wage, price=price,
                         initial_net_worth=initial_net_worth)  # fill the init entries

        # -- Capital goods market -- #
        self.pre_shock_prod = None
        trade_cost = transport_cost
        self.market_share = [1 / self.model.num_firms1,
                             1 / self.model.num_firms1]
        # self.market_share_history = []
        self.regional_orders = []  # By consumption good firms (in own region)
        self.export_orders = []  # By consumption good firms (not own region)
        self.orders_filled = None
        # self.demands = [0, 0]
        self.brochure_regional = [self.productivity[0],
                                  self.price,
                                  self.unique_id]
        self.brochure_export = [self.productivity[0],
                                self.price * (1 + trade_cost),
                                self.unique_id]
        self.profits = 0
        self.real_demand_cap = [0, 0]
        self.past_demand = self.real_demand_cap
        self.production_made = 0
        self.client_IDs = []
        # self.lifecycle = 0
        self.RD_budget = 0
        self.IM = 0
        self.IN = 0
        self.productivity_list = []
        self.bankrupt = False
        self.total_costs_cap = None

    def RD(self):
        """Research and development: Productivity.
           TODO: write more extensive description.
        """
        # -- DETERMINE RD BUDGET -- #
        # Split budget (if any) between innovation (IN) and imitation (IM)
        # --------
        if self.sales > 0:
            _, self.IN, self.IM = rd.calculateRDBudget(self.sales,
                                                       self.net_worth)
            self.RD_budget = self.IN + self.IM

        # -- RD parameters -- #
        # --------
        # COMMENT:
        # TODO: as input parameters?
        # --------
        IN = self.IN  # Budget for innovation
        prod = self.productivity
        Z = 0.3  # Bernoulli exp parameter
        a = 2  # Beta distribution: alpha
        b = 4  # Beta distribution: beta
        x_low = -0.05  # Beta distribution: upper bound
        x_up = 0.05  # Beta distribution: lower bound
        in_productivity = [0, 0]

        # -- INNOVATION -- #
        # Bernoulli draw to determine success of innovation
        if bernoulli.rvs(1 - exp(-Z * IN)) == 1:
            # --------
            # COMMENT:
            # (now there is 1 bernoulli for production and machine
            # productivity, but can be changed)
            # If this stays the same, 2 numbers can be drawn together?
            # --------
            # New production productivity (B) from innovation
            a_0 = (1 + x_low + beta.rvs(a, b) * (x_up - x_low))
            in_productivity[1] = prod[0] * a_0
            # New machine productivity (A) from innovation
            a_1 = (1 + x_low + beta.rvs(a, b) * (x_up - x_low))
            in_productivity[0] = prod[1] * a_1

        # -- IMITATION -- #
        IM = self.IM  # Imitation budget
        firm_ids = self.model.ids_firms1  # list of ids
        agents = self.model.firms1  # list of agents (objects)
        reg = self.region
        Z = 0.3
        e = 5  # geographical distance for imitation
        im_productivity = [0, 0]

        # Bernoulli draw to determine success of imitation
        if bernoulli.rvs(1 - exp(-Z * IM)) == 1:
            # Store imitation probabilities and the corresponding firms
            # --------
            # COMMENT: TODO: change to dict
            # --------
            IM_prob = []
            IM_prob_id = []
            # For all capital-good firms, compute inverse Euclidean
            # technological distances
            # --------
            # COMMENT: change this to vector operation instead of for loop
            # --------
            for firm_id in range(len(firm_ids)):
                firm = agents[firm_id]
                distance = (sqrt(pow(prod[0] - firm.productivity[0], 2) +
                                 pow(prod[0] - firm.productivity[0], 2)))
                if distance == 0:
                    IM_prob.append(0)
                elif firm.region != reg:
                    # Increase distance if firm is in other region
                    # (add geopgraphical distance)
                    # --------
                    # COMMENT:
                    # can be removed, but region might be added again
                    # --------
                    IM_prob.append(1 / e * distance)
                else:
                    IM_prob.append(1 / distance)
                # COMMENT: this contains al firm IDs??
                IM_prob_id.append(firm.unique_id)

            # Cumulative probability
            sum_prob = sum(IM_prob)
            if (sum_prob > 0):
                acc = 0
                for i in range(len(IM_prob)):
                    # --------
                    # COMMENT: change to norm (vector norm) ?
                    # --------
                    acc += IM_prob[i] / sum_prob
                    IM_prob[i] = acc
                # Randomly pick a firm to imitate (index j)
                rnd = uniform(0, 1)
                j = bisect.bisect_right(IM_prob, rnd)
                # Copy that firm's technology
                if j < len(IM_prob):
                    # firm = agents[IM_prob_id[j]]
                    firm = agents[j]
                    im_productivity[0] = firm.productivity[0]
                    im_productivity[1] = firm.productivity[1]

        # Recovering lab productivity after disaster
        if self.pre_shock_prod != 0:
            # If there was a shock, firm has recorded pre-shock productivity
            self.productivity[1] = self.pre_shock_prod
            # set to zero otherwise every period the firm will do this
            self.pre_shock_prod = 0

        # -- ADOPTING NEW TECHNOLOGY -- #
        # # Track productivity options over time
        # --------
        # COMMENT: can be removed?
        # --------
        # self.productivity_list.append([self.productivity[0],
        #                                in_productivity[0],
        #                                im_productivity[0]])
        # --------
        # COMMENT: Why rounding?
        # --------
        self.productivity[0] = round(max(self.productivity[0],
                                         in_productivity[0],
                                         im_productivity[0], 1), 3)
        self.productivity[1] = round(max(self.productivity[1],
                                         in_productivity[1],
                                         im_productivity[1], 1), 3)

    def calculateProductionCost(self):
        """Calculates the unit cost of production. """

        # # This is for CCA so it is off at the moment
        # if self.flooded:
        #     damages = min(self.model.S/self.CCA_resilience[0],
        #                   self.model.S)

        if self.damage_coeff > 0:
            # Store pre-flood productivity
            self.pre_shock_prod = self.productivity[1]
            self.productivity[1] = cd.destroy_fraction(self.productivity[1],
                                                       self.damage_coeff)
        self.cost = self.wage / self.productivity[1]

    def calculatePrice(self, markup=0.04):
        """Calculate unit price. """
        self.price = 1 + markup * self.cost

    def advertise(self):
        """Advertise products to consumption-good firms. """

        # Create brochures
        # --------
        # COMMENT: convert to dicts
        # --------
        self.brochure_regional = [self.productivity[0],
                                  self.price,
                                  self.unique_id]
        trade_cost = self.model.governments[0].transport_cost
        self.brochure_export = [self.productivity[0],
                                self.price * (1 + trade_cost),
                                self.unique_id]

        # Choose potential clients (PC) to advertise to
        r = self.region
        total_pool = self.model.ids_cons_serv_region0
        # # With regions:
        # if r == 0:
        #     total_pool = self.model.ids_cons_serv_region0
        #     total_pool += self.model.sample_client_inland
        # elif r == 1:
        #     total_pool = (self.model.ids_cons_serv_region1 +
        #                   self.model.sample_client_coastal)

        # Sample new client firms
        # COMMENT: change 1 to 0
        if len(self.client_IDs) > 1:
            new_clients = 1 + round(len(self.client_IDs) * 0.2)
        else:
            new_clients = 10
        new_sampled_clients = sample(total_pool, new_clients)

        # --------
        # COMMENT: change client_IDs list --> dict --> list structure
        # --------
        self.client_IDs = self.client_IDs + new_sampled_clients
        self.client_IDs = dict.fromkeys(self.client_IDs)

        # Send brochure to chosen firms
        for firm_id in self.client_IDs:
            client = self.model.schedule.agents[firm_id]
            if client.region == self.region:
                client.offers.append(self.brochure_regional)
            elif client.region == 1 - r:
                client.offers.append(self.brochure_export)

        # Note: return two lists of ids, one for local and one for export?
        self.client_IDs = list(self.client_IDs)
        return self.client_IDs

    def wage_determination(self):
        """Calculate individual wage as the maximum of the minimum wage
           in the region and the maximum wage in the region.
        """
        gov = self.model.governments[0]
        # Get minimum wage in this region (determined by government)
        minimum_wage = gov.min_wage[self.region]

        # Get consumption firm top wages in region
        # TODO include datacollector object
        top_wages = self.model.datacollector.model_vars["Top_wage"]
        top_wage = top_wages[int(self.model.schedule.time)][self.region]
        # Set wage to max of min wage and top paying wage
        self.wage = max(minimum_wage, top_wage)

    def hire_and_fire(self):
        """Open vacancies and fire employees based on demand.

        COMMENT: also check hire_and_fire function in labor_dynamics.py
                 and whether it is necessary to return all arguments
                 and save them all in firm object.
        """
        # --------
        # COMMENT: move first part to separate function? --> handles demand
        # --------
        self.past_demand = self.real_demand_cap
        self.real_demand_cap = [0, 0]
        if self.regional_orders:
            demand_int = 0
            for order in self.regional_orders:
                demand_int += order[0]
            self.real_demand_cap[0] = demand_int
        if self.export_orders:
            demand_exp = 0
            for order in self.export_orders:
                demand_exp += order[0]
            self.real_demand_cap[1] = demand_exp

        self.labor_demand = (sum(self.real_demand_cap) /
                             self.productivity[1] +
                             (self.RD_budget / self.wage))

        employee_info = ld.hire_and_fire(self.labor_demand,
                                         self.employees_IDs,
                                         self.open_vacancies,
                                         self.model, self.profits, self.wage)
        self.desired_employees = employee_info[0]
        self.employees_IDs = employee_info[1]
        self.open_vacancies = employee_info[2]

    def accounting_orders(self):
        """Checks if firm has satisfied all orders, otherwise the remaining
           orders have to be canceled.
        """
        total_orders = sum(self.real_demand_cap)
        # --------
        # COMMENT: rewrite this function to more logical comparison
        # --------
        self.production_made = max(0, round(min(len(self.employees_IDs),
                                                total_orders /
                                                self.productivity[1]) *
                                            self.productivity[1]))
        # # TODO: write comment
        # if self.damage_coeff > 0:
        #     self.production_made = cd.destroy_fraction(self.production_made,
        #                                                self.damage_coeff)
        # if self.model.schedule.time == self.model.shock_time:
        #     damages = min(self.model.S / self.CCA_resilience[1],
        #                   self.model.S)
        #     self.production_made = (1 - damages) * self.production_made

        # Cancel orders if necessary
        self.orders_filled = min(total_orders, self.production_made)
        if self.orders_filled < total_orders:
            # Regional orders are processed (canceled?) before export orders
            orders = self.regional_orders + self.export_orders
            amount_to_cancel = total_orders - self.orders_filled
            shuffle(orders)

            # --------
            # COMMENT: change dynamic updating of these lists
            #          (and change to dicts)
            # --------
            while amount_to_cancel > 0 and len(orders) > 0:
                # Firm is constrained by its production
                # Ensure that the correct amount gets canceled from each order
                c = min(orders[0][0], amount_to_cancel)
                orders[0][0] -= c
                buyer = self.model.schedule.agents[orders[0][1]]

                # Order is fully canceled: delete order
                if orders[0][0] <= 0:
                    buyer.order_canceled = True
                    del orders[0]
                # Order is partially canceled: reduce ordered amount
                else:
                    buyer.order_reduced = c
                amount_to_cancel -= c

        # -- ACCOUNTING -- #
        self.sales = self.orders_filled * self.price
        self.total_costs_cap = self.cost * self.orders_filled
        self.profits = self.sales - self.total_costs_cap - self.RD_budget
        # self.profits = (self.sales - self.total_costs_cap - self.RD_budget -
        #                 self.CCA_RD_budget)
        self.net_worth += self.profits
        # --------
        # COMMENT: Lists below are only used to process orders, no need to
        #          assign them as attributes of the firm objects?
        # --------
        self.regional_orders = []
        self.export_orders = []

    def market_share_normalized(self):
        """Update my market share so that it's normalized.
           The government does it at central level so the firm
           just retrieve its value (linked to its unique_id).

        COMMENT: not really necessary to have separate function only
                 for retrieving these values
        """
        gov = self.model.governments[0]
        market_share = gov.market_shares_normalized_cap[self.unique_id]
        # --------
        # COMMENT: why round? why different rounding for other firms?
        # --------
        self.market_share = [round(i, 6) for i in market_share]

    def stage0(self):
        """Stage 0:
           TODO: write short description for all stages
        """

        # # CCA DYNAMICS
        # if self.region == 0:
        #     self.CCA_RD()
        #     self.migration = False

        # if self.lifecycle > 16:  # or self.new_born == True:
        #     migr_prob = migration.migration_probability(self.real_demand_cap,
        #                                                 self.past_demand,
        #                                                 self.region)
        #     if migr_prob > 0:
        #         new_attr = migration.firm_migrate(self,
        #                                                migr_prob,
        #                                                self.model,
        #                                                self.region,
        #                                                self.unique_id,
        #                                                self.employees_IDs,
        #                                                self.net_worth,
        #                                                self.wage)
        #         self.region = new_attr[0]
        #         self.employees_IDs = new_attr[1]
        #         self.net_worth = new_attr[2]
        #         self.wage = new_attr[3]
        # if self.lifecycle > 4:
        #     if self.region == 0:
        #         clients = self.model.ids_cons_serv_region0
        #     else:
        #         clients = self.model.ids_cons_serv_region1

        #     if (self.model.governments[0].aggregate_employment[self.region]
        #         == 0) or len(clients) == 0:
        #         new_attr = migration.firm_migrate(self, 1, self.model,
        #                                                self.region,
        #                                                self.unique_id,
        #                                                self.employees_IDs,
        #                                                max(0,self.net_worth),
        #                                                self.wage)
        #         self.region = new_attr[0]
        #         self.employees_IDs = new_attr[1]
        #         self.net_worth = new_attr[2]
        #         self.wage = new_attr[3]
        #     # if self.sales > 0:

        # Handle flood if it occurs
        if self.model.is_flood_now:
            # TODO: modify damage_coeff here!
            self.damage_coeff = cd.depth_to_damage(self.model.flood_depth,
                                                   self.height, self.type)
        else:
            self.damage_coeff = 0

        self.RD()
        self.calculateProductionCost()
        self.calculatePrice()
        self.advertise()

    def stage1(self):
        """Stage 1:
           TODO: write short description for all stages
        """
        self.wage_determination()
        self.hire_or_fire()  # SL: renamed

    def stage2(self):
        pass

    def stage3(self):
        """Stage 3:
           TODO: write short description for all stages
        """
        households = self.model.schedule.agents_by_type["Household"]
        ld.update_employees_wage(self.employees_IDs, households, self.wage)
        self.accounting_orders()

    def stage4(self):
        pass

    def stage5(self):
        pass

    def stage6(self):
        """Stage 6:
           TODO: write short description for all stages
        """
        self.market_share_normalized()

        # if self.lifecycle > 7:
        #     self.new_born = False

        if self.lifecycle > 2:
            if len(self.employees_IDs) > 1:  # and self.model.time > 50:
                self.subsidiary = ac.new_entry(self.profits,
                                               len(self.employees_IDs),
                                               self.subsidiary, self.wage)
                if self.subsidiary > 3:  # and uniform(0, 1) > 0.5:
                    if self.region == 0:
                        # self.model.governments[0].entrants_cap_coastal +=1
                        self.model.subs_cap_coastal.append(self)
                    elif self.region == 1:
                        # self.model.governments[0].entrants_cons_inland +=1
                        self.model.subs_cons_inland.append(self)
                    self.subsidiary = 0

            # TODO: comment
            if self.net_worth <= 0 and sum(self.real_demand_cap) < 1:
                if len(self.model.ids_firms1) > 10:
                    self.model.kill_cap_coastal.append(self)

                # Fire employees
                # COMMENT: use function in accounting file.
                for employee_id in self.employees_IDs:
                    employee = self.model.schedule.agents[employee_id]
                    employee.employer_ID = None
                self.employees_IDs = []

        self.lifecycle += 1


class ConsumptionGoodFirm(ConsumptionFirm):
    firm_type = 'conG'

    def __init__(self,
                 unique_id: str,
                 model: model_init.RHuCC_Model,
                 productivity: float,
                 location: str,
                 wage: float,
                 price: float,
                 initial_net_worth: float,
                 initial_capital_output_ratio: float,
                 initial_machines: int,
                 initial_amount_capital: float,  # dropped initial_number_of_machines, possibly redundant
                 flood_exp: tuple[float],
                 flood_dmg: tuple[float],
                 ):
        super(ConsumptionGoodFirm, self).__init__(
            unique_id=unique_id,
            model=model,
            productivity=productivity,
            location=location,
            wage=wage,
            price=price,
            initial_net_worth=initial_net_worth,
            initial_capital_output_ratio=initial_capital_output_ratio,
            initial_machines=initial_machines,
            initial_amount_capital=initial_amount_capital,
            flood_exp=flood_exp,
            flood_dmg=flood_dmg
        )

        self.market_share = [1 / self.model.num_firms2,
                             1 / self.model.num_firms2,
                             1 / self.model.num_firms2]


class ConsumptionServiceFirm(ConsumptionFirm):
    firm_type = 'conS'

    def __init__(self,
                 unique_id: str,
                 model: model_init.RHuCC_Model,
                 productivity: float,
                 location: str,
                 wage: float,
                 price: float,
                 initial_net_worth: float,
                 initial_capital_output_ratio: float,
                 initial_machines: int,
                 initial_amount_capital: float,  # dropped initial_number_of_machines, possibly redundant
                 flood_exp: tuple[float],
                 flood_dmg: tuple[float]
                 ):
        super(ConsumptionServiceFirm, self).__init__(
            unique_id=unique_id,
            model=model,
            productivity=productivity,
            location=location,
            wage=wage,
            price=price,
            initial_net_worth=initial_net_worth,
            initial_capital_output_ratio=initial_capital_output_ratio,
            initial_machines=initial_machines,
            initial_amount_capital=initial_amount_capital,
            flood_exp=flood_exp,
            flood_dmg=flood_dmg
        )
        self.market_share = [1 / self.model.num_firms3,
                             1 / self.model.num_firms3,
                             1 / self.model.num_firms3]
