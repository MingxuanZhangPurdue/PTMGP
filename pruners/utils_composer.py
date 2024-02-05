import re
from typing import Union
from composer.core import State, Time
from composer.optim.scheduler import ComposerScheduler, _convert_time

class RelativeLinearScheduler(ComposerScheduler):
    def __init__(self,
                 t_start: Union[str, Time] = '0dur',
                 t_end: Union[str, Time] = '1dur', 
                 alpha_i: float = 1.0, 
                 alpha_f: float = 0.0, 
                 ):
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.t_end = Time.from_timestring(t_end) if isinstance(t_end, str) else t_end
        self.t_start = Time.from_timestring(t_start) if isinstance(t_start, str) else t_start
        assert t_start.unit == t_end.unit, \
            f"t_start and t_end must have the same unit, but got {t_start.unit} and {t_end.unit}"
        assert t_start.value < t_end.value, \
            f"t_start must be < t_end, but got {t_start.value} and {t_end.value}"

    def __call__(self, state: State):
        t_start = _convert_time(self.t_start, state)
        t_end = _convert_time(self.t_end, state)
        current_time = state.timestamp.get(t_start.unit)
        frac_of_total = min(1.0, ((current_time - t_start)/(t_end - t_start)).value)
        assert 0.0 <= frac_of_total <= 1.0, \
            f"frac_of_total must be between 0 and 1, but got {frac_of_total}, this may be due to current_time being outside of the range [t_start, t_end]"
        current_factor = self.alpha_i + frac_of_total * (self.alpha_f - self.alpha_i)
        return current_factor

class LinearWithRewindsScheduler(ComposerScheduler):
    def __init__(self,
                 rewind_start: Union[str, Time],
                 rewind_interval: Union[str, Time],
                 num_rewinds: int = 1,
                 alpha_i: float = 1.0,
                 alpha_f: float = 0.0,
                 alpha_i_rewind: float = 1.0,
                 alpha_f_rewind: float = 0.0,):
        
        assert num_rewinds >= 1, "num_rewinds must be >= 1"
        rewind_start = Time.from_timestring(rewind_start) if isinstance(rewind_start, str) else rewind_start
        rewind_interval = Time.from_timestring(rewind_interval) if isinstance(rewind_interval, str) else rewind_interval
        assert rewind_start.unit == rewind_interval.unit, \
            f"rewind_start and rewind_interval must have the same unit, but got {rewind_start.unit} and {rewind_interval.unit}"
        assert rewind_start.value > 0, \
            f"rewind_start must be > 0, but got {rewind_start.value}"
        self.schedulers = [RelativeLinearScheduler(t_start=0*rewind_start,
                                                   t_end=rewind_start, 
                                                   alpha_i=alpha_i, 
                                                   alpha_f=alpha_f)]
        for i in range(num_rewinds):
            self.schedulers.append(RelativeLinearScheduler(t_start=rewind_start + i*rewind_interval,
                                                           t_end=rewind_start + (i+1)*rewind_interval, 
                                                           alpha_i=alpha_i_rewind, 
                                                           alpha_f=alpha_f_rewind))
        self.num_rewinds = num_rewinds
        self.rewind_start = rewind_start
        self.rewind_interval = rewind_interval

        self.current_scheduler_index = 0
        self.next_rewind_time = self.rewind_start

    def __call__(self, state: State):
        next_rewind_time = _convert_time(self.next_rewind_time, state)
        if state.timestamp < next_rewind_time:
            return self.schedulers[self.current_scheduler_index](state)
        else:
            if self.current_scheduler_index+1 > self.num_rewinds:
                return self.schedulers[self.current_scheduler_index](state)
            else:
                self.current_scheduler_index += 1
                self.next_rewind_time += self.rewind_interval
                return self.schedulers[self.current_scheduler_index](state)

def _get_unit_and_value(time):
    time_units = ["ep", "ba", "dur"]
    # regex for parsing time string, matches timeunit and chars prior to unit as value
    _TIME_STR_REGEX = re.compile(r'^(.+)(' + r'|'.join([fr'{time_unit}' for time_unit in time_units]) + r')$',
                                flags=re.IGNORECASE)
    match = _TIME_STR_REGEX.findall(time)
    if len(match) != 1:
        raise ValueError(f'Invalid time string: {time}')
    match = match[0]
    match = [x for x in match if x != '']
    assert len(match) == 2, 'each match should have a number followed by the key'
    value = match[0]
    unit = match[1]
    value = float(value)  # always parsing first as float b/c it could be scientific notation
    if unit == "ba":
        if int(value) != value:
            raise TypeError(f'value {value} is not an integer. Units {unit} require integer values.')
        value = int(value)
    return unit, value

def _convert_timestr_to_int(time, max_train_steps, train_dataloader_len):
    if isinstance(time, int):
        return time
    elif isinstance(time, str):
        unit, value = _get_unit_and_value(time)
        if unit.casefold() == "dur".casefold():
            return int(value*max_train_steps)
        elif unit.casefold() == "ba".casefold():
            return int(value)
        else:
            return int(value*train_dataloader_len)
    else:
        raise ValueError("time must be either int or str.")