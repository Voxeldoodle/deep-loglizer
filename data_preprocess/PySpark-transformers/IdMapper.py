from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, IntegerType
import pyspark.sql.functions as F
from pyspark.sql import Window

 
class IdMapper(
                    Transformer,               # Base class
                    HasInputCol,               # Sets up an inputCol parameter
                    HasOutputCol,              # Sets up an outputCol parameter
                    DefaultParamsReadable,     # Makes parameters readable from file
                    DefaultParamsWritable      # Makes parameters writable from file
                    ):
    """
    Custom Transformer for splitting input data in temporal chunks
    """

    mapping = Param(
        Params._dummy(),
        "mapping",
        "Dictionary mapping each value with its id"
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, mapping=None):
        """
        Constructor: set values for all Param objects
        """
        super().__init__()
        self._setDefault(mapping=None)
        self._setDefault(outputCol="ID")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, mapping=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
  
    def setMapping(self, new_mapping):
        return self.setParams(mapping=new_mapping)
  
    # Required if you use Spark >= 3.0
    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)
  
    # Required if you use Spark >= 3.0
    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol)

    def getMapping(self):
        return self.getOrDefault(self.mapping)
  
    def _transform(self, df: DataFrame):
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        mapping = self.getMapping()
        
        if mapping is None:
          mapping = {}
          counter = 1
        elif len(mapping) == 0:
          counter = 1
        else:
          counter = max(mapping.values()) + 1

        idCol = "LineId"
        tmp = df.withColumn(idCol,F.col(idCol).cast(IntegerType())) \
                .groupBy(input_col).min(idCol) \
                .orderBy(f"min({idCol})")

        for row in tmp.collect():
          mapping[row["EventTemplate"]] = counter
          counter+=1

        mapping_func = lambda x: mapping.get(x) 
        df = df.withColumn(output_col,F.udf(mapping_func, IntegerType())(input_col))

        return df