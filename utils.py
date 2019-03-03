import time

def backspace():
    print('\r', end='')                     # use '\r' to go back

def draw_progess_bar(n_finished, n_jobs, bar_length=30, sleep_time=0.0):
    finish_percent = int(float((n_finished)) / n_jobs * 100)
    progress_length = int(finish_percent * bar_length /100)
    print ("[%s>%s] %d%%" % ('=' * progress_length, ' ' * (bar_length - progress_length), finish_percent), end='')
    backspace()

    time.sleep(sleep_time)
