# Ref: https://gist.github.com/cukie/fcab1cd5a321d70a219e
from typing import Callable


class ReusableGenerator(object):
    """A decorator that allows a generator to become reusable.
    That is, every time you call the __iter__ method on a decorated generator, you'll
    get a fresh instance of the generator.

    Inspiration from: http://stackoverflow.com/questions/1376438/how-to-make-a-repeating-generator-in-python

    Usage:

        @ReusableGenerator
        def testGen(x):
            for i in xrange(x):
                yield i

        a = testGen(10)
        list(a)  # [1,2,3,4,5,6,7,8,9]
        list(a)  # [1,2,3,4,5,6,7,8,9]

    """

    def __init__(self, func: Callable):
        self.args = ()
        self.kwargs = {}
        self.func = func

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def __iter__(self):
        return self.func(*self.args, **self.kwargs)
