from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
from pyspark.sql import Window

 
class Windower(
                    Transformer,               # Base class
                    HasInputCol,               # Sets up an inputCol parameter
                    HasOutputCol,              # Sets up an outputCol parameter
                    DefaultParamsReadable,     # Makes parameters readable from file
                    DefaultParamsWritable      # Makes parameters writable from file
                    ):
    """
    Custom Transformer for splitting input data in temporal chunks
    """

    window_size = Param(
        Params._dummy(),
        "window_size",
        "Number of elements per window",
        typeConverter=TypeConverters.toInt,
    )
    stride = Param(
        Params._dummy(),
        "stride",
        "Number of elements to jump between windows",
        typeConverter=TypeConverters.toInt,
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, window_size=None, stride=None):
        """
        Constructor: set values for all Param objects
        """
        super().__init__()
        self._setDefault(window_size=10)
        self._setDefault(stride=1)
        # self._setDefault(outputCol="SessionID")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, window_size=None, stride=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
  
    def setWindowSize(self, new_window_size):
        return self.setParams(window_size=new_window_size)
    
    def setStride(self, new_stride):
        return self.setParams(stride=new_stride)
  
    # Required if you use Spark >= 3.0
    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)
  
    # Required if you use Spark >= 3.0
    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol)

    def getWindowSize(self):
        return self.getOrDefault(self.window_size)
    
    def getStride(self):
        return self.getOrDefault(self.stride)
  
    def _transform(self, df: DataFrame):
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        size = self.getWindowSize()
        stride = self.getStride()
        id_padding = 0
        label_padding = 0

        # windows
        w = Window.partitionBy('group')
        w_ordered = Window.partitionBy('group').orderBy(input_col)
        w_ordered_limited = Window.partitionBy('group').orderBy(input_col).rowsBetween(0, size - 1)

        if stride == 1:
          filter_cond = (F.col('n') >= size) & (F.size('Windows') == size)
        else:
          filter_cond = (F.col('n') >= size) & (F.col('n_row') < size) & (F.col('n_row') % stride == 1) & (F.size('Windows') == size)

        if size % stride != 0:
          transf = lambda dfcol : F.slice(F.concat(dfcol, F.array_repeat(F.lit(id_padding), size - 1)), 1, size)
        else:
          transf = lambda dfcol : F.when(F.col('n') < size, F.slice(F.concat(dfcol, F.array_repeat(F.lit(id_padding), size - 1)), 1, size)) \
                                  .otherwise(F.col(dfcol))
        df = df.select(
          'group',
          F.collect_list(input_col).over(w_ordered_limited).alias('Windows'),
          F.collect_list('Label').over(w_ordered_limited).alias('Groups'),
          F.count('group').over(w).alias('n'),
          F.row_number().over(w_ordered).alias('n_row')
          ) \
          .withColumn('Windows', transf("Windows")) \
          .withColumn('Groups',  transf("Groups")) \
          .filter( ((F.col('n') < size) & (F.col('n_row') == 1))
                  | (filter_cond)) \
          .drop('n', 'n_row')

        return df