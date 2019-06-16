package com.UpGrad.Data_Pre_Processing;


import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.MaxAbsScaler;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class AutoDataPipeline {

	public static void main(String[] args) {

	Logger.getLogger("org").setLevel(Level.ERROR);
	Logger.getLogger("akka").setLevel(Level.ERROR);

	SparkSession sparkSession = SparkSession.builder()
			.appName("SparkSQL")
			.master("local[*]")
			.getOrCreate();

	Dataset<Row> df1 = sparkSession.read().option("header", true).option("inferschema", true).csv("data/auto-miles-per-gallon-Raw.csv");
	//Reading Data from a CSV file //Inferring Schema and Setting Header as True


	//****************************************** Handling missing values ******************************************************************

	//Casting MPG and HORSEPOWER from String to Double
	Dataset<Row> df2 = df1.selectExpr("cast(MPG as double ) MPG", "CYLINDERS","DISPLACEMENT",
			"cast(HORSEPOWER as double) HORSEPOWER","WEIGHT", 
			"ACCELERATION","MODELYEAR","NAME");

	//******************************************Replace missing values with approximate mean values*************************************

	//Imputer method automatically replaces null values with mean values.
	Imputer imputer = new Imputer()
			.setInputCols(new String[]{"MPG","HORSEPOWER"})
			.setOutputCols(new String[]{"MPG-Out","HORSEPOWER-Out"});


	//****************************************Assembling the Vector and Label************************************************************

	//Assembling the features in the dataFrame as Dense Vector
	VectorAssembler assembler = new VectorAssembler()
			.setInputCols(new String[]{"CYLINDERS","WEIGHT","HORSEPOWER-Out","DISPLACEMENT"})
			.setOutputCol("features");

	//*********************************************Scaling the Vector***********************************************************************

	//Scaling the features between 0-1
	MaxAbsScaler scaler = new MaxAbsScaler() //Performing MaxAbsScaler() Transformation
			.setInputCol("features")
			.setOutputCol("scaledFeatures");


	//*********************************************Normalizing the Vector*********************************************************************

	//Normalizing the vector. Converts vector to a unit vector
	Normalizer normalizer = new Normalizer() //Performing Normalizer() Transformation
			.setInputCol("scaledFeatures")
			.setOutputCol("normFeatures")
			.setP(2.0);
	
	Pipeline pipeline = new Pipeline().setStages(new PipelineStage [] {imputer,assembler,scaler,normalizer});
	
	PipelineModel model = pipeline.fit(df2);
	Dataset<Row> df3 = model.transform(df2);
	df3.show();
	
	Dataset<Row> df4= df3.select("MPG-Out","normFeatures");
	df4.show();
}
}

