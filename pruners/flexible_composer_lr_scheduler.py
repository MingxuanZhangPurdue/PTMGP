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
        if self.alpha_i == self.alpha_f:
            # alpha_i == alpha_f, implies constant lr scheduler, hence return alpha_i
            return self.alpha_i
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
                 t_iw: Union[str, Time],
                 t_fw: Union[str, Time] = None,
                 t_rewind: Union[str, Time] = None,
                 num_rewinds: int = 0,
                 alpha_i_iw: float = 1.0,
                 alpha_f_iw: float = 0.0,
                 alpha_i_rewind: float = 1.0,
                 alpha_f_rewind: float = 0.0,
                 alpha_i_fw: float = 1.0,
                 alpha_f_fw: float = 0.0,):
        
        if num_rewinds >= 1 and t_rewind is None:
            raise ValueError("t_rewind must be provided when num_rewinds is >= 1, otherwise set num_rewinds to 0.")
        t_iw = Time.from_timestring(t_iw) if isinstance(t_iw, str) else t_iw
        t_rewind = Time.from_timestring(t_rewind) if isinstance(t_rewind, str) else t_rewind
        t_fw = Time.from_timestring(t_fw) if isinstance(t_fw, str) else t_fw
        if t_rewind is not None:
            assert t_rewind.unit == t_iw.unit, \
                f"t_rewind and t_iw must have the same unit, but got {t_rewind.unit} and {t_iw.unit}"
            assert t_rewind.value > 0, \
                f"t_rewind.value must be > 0, but got {t_rewind.value}."
        if t_fw is not None:
            assert t_fw.unit == t_iw.unit, \
                f"t_fw and t_iw must have the same unit, but got {t_fw.unit} and {t_iw.unit}"
            assert t_fw.value > 0, \
                f"t_fw.value must be > 0, but got {t_fw.value}."
        assert t_iw.value > 0, \
            f"t_iw.value must be > 0, but got {t_iw.value}."
        self.schedulers = [
            RelativeLinearScheduler(
                t_start=0*t_iw,
                t_end=t_iw, 
                alpha_i=alpha_i_iw,
                alpha_f=alpha_f_iw
            )
        ]
        if alpha_i_rewind== alpha_f_rewind:
            assert num_rewinds == 1, "alpha_i_rewind and alpha_f_rewind are the same, which implies constant lr scheduler during rewind, hence num_rewinds must be 1"
        for i in range(num_rewinds):
            self.schedulers.append(
                RelativeLinearScheduler(
                    t_start=t_iw + i*t_rewind,
                    t_end=t_iw + (i+1)*t_rewind, 
                    alpha_i=alpha_i_rewind, 
                    alpha_f=alpha_f_rewind)
                )
        if t_fw is not None:
            self.schedulers.append(
                RelativeLinearScheduler(
                    t_start=t_iw + num_rewinds*t_rewind,
                    t_end=t_iw + num_rewinds*t_rewind + t_fw,
                    alpha_i=alpha_i_fw,
                    alpha_f=alpha_f_fw
                )
            )

        self.num_rewinds = num_rewinds
        self.t_iw = t_iw
        self.t_rewind = t_rewind if t_rewind is not None else 0*t_iw
        self.t_fw = t_fw if t_fw is not None else 0*t_iw

        self.initial_warmup_end = t_iw
        self.current_scheduler_index = 0
        self.next_rewind_time = t_iw
        self.final_warmup_start = t_iw + num_rewinds*t_rewind

    def __call__(self, state: State):
        initial_warmup_end = _convert_time(self.initial_warmup_end, state)
        final_warmup_start = _convert_time(self.final_warmup_start, state)
        if state.timestamp < initial_warmup_end:
            return self.schedulers[0](state)
        elif state.timestamp >= final_warmup_start:
            return self.schedulers[-1](state)
        else:
            next_rewind_time = _convert_time(self.next_rewind_time, state)
            if state.timestamp == next_rewind_time:
                self.current_scheduler_index += 1
                self.next_rewind_time += self.t_rewind
            return self.schedulers[self.current_scheduler_index](state)