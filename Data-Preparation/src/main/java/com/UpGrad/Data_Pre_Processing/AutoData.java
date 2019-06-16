package com.UpGrad.Data_Pre_Processing;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.ImputerModel;
import org.apache.spark.ml.feature.MaxAbsScaler;
import org.apache.spark.ml.feature.MaxAbsScalerModel;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.Row;
import static org.apache.spark.sql.functions.col;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.spark_project.dmg.pmml.Delimiter;

public class AutoData {

	public static void main(String[] args) {

		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);


		SparkSession sparkSession = SparkSession.builder()
				.appName("SparkML")
				.master("local[*]")
				.getOrCreate();


		Dataset<Row> df1 = sparkSession.read().option("header", true).option("inferschema", false).csv("data/auto-miles-per-gallon-Raw.csv");

		//Reading Data from a CSV file //Inferring Schema and Setting Header as True

		df1.show(); //Displaying samples
		df1.printSchema(); //Printing Schema
		df1.describe().show(); //Statistically summarizing about the data


		//****************************************** Handling missing values ******************************************************************

		//Casting MPG and HORSEPOWER from String to Double
		Dataset<Row> df2 = df1.select(col("MPG").cast("Double"), col("CYLINDERS"),col("DISPLACEMENT"),
				col("HORSEPOWER").cast("Double"),col("WEIGHT"), 
				col("ACCELERATION"),col("MODELYEAR"),col("NAME"));
		
		System.out.println("*************************Casting columns********************************");
		df2.show(); //Displaying samples 
		df2.printSchema(); //Printing new Schema

		//Removing Rows with missing values
		System.out.println("********************Removing records with missing values**********************");
		Dataset<Row> df3 = df2.na().drop(); //Dataframe.na.drop removes any row with a NULL value
		df3.describe().show(); //Describing DataFrame

		//******************************************Replace missing values with approximate mean values*************************************

		System.out.println("*******************Replacing records with missing values********************");
		
		//Imputer method automatically replaces null values with mean values.
		Imputer imputer = new Imputer()
				.setInputCols(new String[]{"MPG","HORSEPOWER"})
				.setOutputCols(new String[]{"MPG-Out","HORSEPOWER-Out"});

		ImputerModel imputeModel = imputer.fit(df2); //Fitting DataFrame into a model
		Dataset<Row> df4=imputeModel.transform(df2); //Transforming the DataFrame
		df4.show();
		df4.describe().show(); //Describing the dataframe

		
		//Removing unnecessary columns
		Dataset<Row> df5 =df4.drop(new String[] {"MPG","HORSEPOWER"});

		//*******************************************Statistical Data Analysis*************************************************************

		System.out.println("***********************Performing statistical exploration*********************");
		
		StructType autoSchema = df5.schema(); //Inferring Schema
		
		for ( StructField field : autoSchema.fields() ) {	//Running through each column and performing Correlation Analysis
			if ( ! field.dataType().equals(DataTypes.StringType)) {
				System.out.println( "Correlation between MPG-Out and " + field.name()
				+ " = " + df5.stat().corr("MPG-Out", field.name()) );
			}
		}


		//****************************************Assembling the Vector and Label************************************************************

		System.out.println("******************************Assembling the vector************************");
		//Renaming MPG-Out as lablel
		Dataset<Row> df6= df5.select(col("MPG-Out").as("label"),col("CYLINDERS"),col("WEIGHT"),col("HORSEPOWER-Out"),col("DISPLACEMENT"));

		//Assembling the features in the dataFrame as Dense Vector
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"CYLINDERS","WEIGHT","HORSEPOWER-Out","DISPLACEMENT"})
				.setOutputCol("features");

		Dataset<Row> df7 = assembler.transform(df6).select("label","features");	
		df7.show();

		//*********************************************Scaling the Vector***********************************************************************

	
		//Scaling the features between 0-1
		MaxAbsScaler scaler = new MaxAbsScaler() //Performing MaxAbsScaler() Transformation
				.setInputCol("features")
				.setOutputCol("scaledFeatures");

		// Building and Fitting in a MaxAbsScaler Model
		MaxAbsScalerModel scalerModel = scaler.fit(df7); 

		// Re-scale each feature to range [0, 1].
		Dataset<Row> scaledData = scalerModel.transform(df7);

		//*********************************************Normalizing the Vector*********************************************************************

		System.out.println("**********************************Scaling and Normalizing the vector***************************");	
		//Normalizing the vector. Converts vector to a unit vector
		Normalizer normalizer = new Normalizer() //Performing Normalizer() Transformation
				.setInputCol("scaledFeatures")
				.setOutputCol("normFeatures")
				.setP(2.0);

		Dataset<Row> NormData = normalizer.transform(scaledData);
		NormData.show();

	}
}
