import re
from typing import Union

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