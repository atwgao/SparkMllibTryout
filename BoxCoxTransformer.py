from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer
#from pyspark.ml.param import Param
from pyspark.ml.param.shared import HasOutputCol, HasInputCol, Param
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType,FloatType

from math import log


class BoxCoxTransformer(Transformer, HasInputCol, HasOutputCol):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, alpha=None):
        super(BoxCoxTransformer, self).__init__()
        self.alpha = Param(self, "alpha", 0)
        self._setDefault(alpha=0)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, alpha=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setAlpha(self, value):
        self._paramMap[self.alpha] = value
        return self

    def getAlpha(self):
        return self.getOrDefault(self.alpha)

    def _transform(self, dataset):
        alpha = self.getAlpha()

        def f(s):
            #print(type(s))
            #print(type(alpha))
            if alpha == 0:
                return log(s)
            elif alpha > 0:
                return (s ** alpha - 1) / alpha

        t = FloatType()
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))
