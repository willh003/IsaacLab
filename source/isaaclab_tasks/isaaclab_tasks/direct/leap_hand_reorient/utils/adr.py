import copy

class LeapHandADR():
    def __init__(self, event_manager, adr_cfg_dict, adr_custom_cfg_dict):
        self.event_manager = event_manager
        self.adr_cfg_dict = adr_cfg_dict
        self.adr_custom_cfg_dict = adr_custom_cfg_dict

        self.adr_cfg_dict_initial = copy.deepcopy(adr_cfg_dict)
        self.save_param_ranges()
        self.increment_counter = 0

    def save_param_ranges(self):
        for term_name, term_params in self.adr_cfg_dict.items():
            if term_name != "num_increments":
                term = self.event_manager.get_term_cfg(term_name)
                for param_name, param_values in term_params.items():
                    self.adr_cfg_dict_initial[term_name][param_name] =\
                        copy.deepcopy(term.params[param_name])

    def print_params(self):
        for term_name, term_params in self.adr_cfg_dict.items():
            if term_name != "num_increments":
                term = self.event_manager.get_term_cfg(term_name)
                print('term_name', term)

    def increase_ranges(self):
        if self.increment_counter >= self.adr_cfg_dict["num_increments"]:
            print('not making a param change')
            return
        else:
            print('making a change')

            self.increment_counter += 1

            for term_name, term_params in self.adr_cfg_dict.items():
                if term_name != "num_increments":
                    term = self.event_manager.get_term_cfg(term_name)
                    for param_name, param_values in term_params.items():
                        lower_limit_inc =\
                            (self.adr_cfg_dict[term_name][param_name][0] -\
                            self.adr_cfg_dict_initial[term_name][param_name][0]
                            ) / float(self.adr_cfg_dict["num_increments"])

                        lower_limit = lower_limit_inc * self.increment_counter +\
                            self.adr_cfg_dict_initial[term_name][param_name][0]

                        upper_limit_inc =\
                            (self.adr_cfg_dict[term_name][param_name][1] -\
                            self.adr_cfg_dict_initial[term_name][param_name][1]
                            ) / float(self.adr_cfg_dict["num_increments"])

                        upper_limit = upper_limit_inc * self.increment_counter +\
                            self.adr_cfg_dict_initial[term_name][param_name][1]

                        new_range = (lower_limit, upper_limit)
                        
                        term.params[param_name] = new_range

    def num_increments(self):
        return self.increment_counter

    def set_num_increments(self, num_increments):
        self.increment_counter = num_increments

    def get_custom_param_value(self, param_group, param_name):
        upper_limit = self.adr_custom_cfg_dict[param_group][param_name][1]
        lower_limit = self.adr_custom_cfg_dict[param_group][param_name][0]
        param_slope = (upper_limit - lower_limit) / float(self.adr_cfg_dict["num_increments"])
        param_value = param_slope * self.increment_counter + lower_limit
        return param_value