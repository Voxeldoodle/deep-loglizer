from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
from pyspark.sql import Window

 
class TemporalPreprocessor(
                    Transformer,               # Base class
                    HasInputCol,               # Sets up an inputCol parameter
                    HasOutputCol,              # Sets up an outputCol parameter
                    DefaultParamsReadable,     # Makes parameters readable from file
                    DefaultParamsWritable      # Makes parameters writable from file
                    ):
    """
    Custom Transformer for splitting input data in temporal chunks
    """

    # interval is a value which we would like to be able to store state for, so we create a parameter.
    interval = Param(
        Params._dummy(),
        "interval",
        "Time in seconds for the length of each group",
        typeConverter=TypeConverters.toInt,
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, interval=None):
        """
        Constructor: set values for all Param objects
        """
        super().__init__()
        self._setDefault(interval=0)
        # self._setDefault(outputCol="SessionID")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, interval=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
  
    def setInterval(self, new_interval):
        return self.setParams(interval=new_interval)
  
    # Required if you use Spark >= 3.0
    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)
  
    # Required if you use Spark >= 3.0
    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol)

    def getInterval(self):
        return self.getOrDefault(self.interval)
  
    def _transform(self, df: DataFrame):
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        interval = self.getInterval()

        w = Window.orderBy(input_col)
        w2 = Window.partitionBy('group1').orderBy(input_col)
        w3 = Window.orderBy('group1', 'group2')

        diff = F.col(input_col).astype('long') - F.lag(F.col(input_col).astype('long')).over(w)
        diff = F.when(diff.isNull(), 0).otherwise(diff)
        diff = (diff > interval).astype('int')

        df = (df
              .withColumn('group1', F.sum(diff).over(w))
              .withColumn('group2', F.floor((F.col(input_col).astype('long') - F.min(F.col(input_col).astype('long')).over(w2)) / interval))
              .withColumn(output_col, F.dense_rank().over(w3))
        )
        df = df.drop('group1', 'group2')

        return df