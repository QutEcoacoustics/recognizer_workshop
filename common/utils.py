import time
import os
from datetime import datetime

log_fn = ''

def time_report (start_time, item_num, num_items, item_type='segment', duration=None, report_every=1, logpath=None):
    """
    Print a message to indicate the progress through items
    :param start_time: what time the whole thing started
    :param item_num: which item we are up to
    :param num_items: how many items total
    :param item_type: what type of item it is (just for a more specific message)
    :param duration: (optional) for audio items, what is the duration of current audio item
    :param report_every: only show the message every this many items
    :return:
    """

    if item_num % report_every == (report_every -1):

        elapsed_time = time.time() - start_time
        time_per_item = elapsed_time / (item_num + 1)
        items_remaining = num_items - item_num - 1
        time_remaining = items_remaining * time_per_item
        mins_remaining = round(time_remaining / 60, 2)

        if duration is None:
            speed_string = ''
        else:
            speed_string = f' ({round(duration / time_per_item, 2)} x realtime)'

        msg = f'average time per {item_type} {round(time_per_item, 2)}s {item_num + 1} in {round(elapsed_time)}s{speed_string}. {item_type}s remaining {items_remaining} about {mins_remaining} min'

        if logpath is None:
            print(msg)
        else:
            mylog(msg, fn=logpath)



def mylog(msg, post='\n', pre='', fn=None):
    global log_fn
    print(msg)

    if fn is None:
        fn = log_fn

    if fn != '':
        f = open(fn, "a")
        f.write(pre + nowstring() + '  ' + msg + post)
        f.close()


def init_log(fn):
    global log_fn
    log_fn = fn


def nowstring():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def fn_append(fn, str):
    """
    appends a string before the extension of a fn
    """
    name, ext = os.path.splitext(fn)
    return f'{name}{str}{ext}'