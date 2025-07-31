from sv_comp.option_sv_comp import OptionsSVComp
from sv_comp.thread_sv_comp import ThreadSVComp

if __name__ == '__main__':
    opt = OptionsSVComp().parse()
    thread = ThreadSVComp(opt)
    thread.forward()
