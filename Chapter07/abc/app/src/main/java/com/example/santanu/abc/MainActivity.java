package com.example.santanu.abc;
import android.content.res.AssetManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.RatingBar;
import android.widget.TextView;
import android.widget.Button;
import android.widget.EditText;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;






public class MainActivity extends AppCompatActivity {

    private TensorFlowInferenceInterface mInferenceInterface;
    private static final String MODEL_FILE = "file:///android_asset/optimized_model.pb";
    private static final String INPUT_NODE = "inputs/X";
    private static final String OUTPUT_NODE = "positive_sentiment_probability";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // Create references to the widget variables

        final TextView desc = (TextView) findViewById(R.id.desc);
        final Button submit = (Button) findViewById(R.id.submit);
        final EditText Review = (EditText) findViewById(R.id.Review);
        final TextView score = (TextView) findViewById(R.id.score);
        final RatingBar ratingBar = (RatingBar) findViewById(R.id.ratingBar);






        //String filePath = "/home/santanu/Downloads/Mobile_App/word2ind.txt";
        final Map<String,Integer> WordToInd = new HashMap<String,Integer>();
        //String line;

        //reader = new BufferedReader(new InputStreamReader(getAssets().open("word2ind.txt")));

        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(getAssets().open("word_ind.txt")));

            // do reading, usually loop until end of file reading
            String line;
            while ((line = reader.readLine()) != null)
            {
                String[] parts = line.split("\n")[0].split(",",2);
                if (parts.length >= 2)
                {

                    String key = parts[0];
                    //System.out.println(key);
                    int value = Integer.parseInt(parts[1]);
                    //System.out.println(value);
                    WordToInd.put(key,value);
                } else

                {
                    //System.out.println("ignoring line: " + line);
                }
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }

        //line = reader.readLine();




        // Create Button Submit Listener

        submit.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                // Read Values
                String reviewInput = Review.getText().toString().trim();
                System.out.println(reviewInput);

                String[] WordVec = reviewInput.replaceAll("[^a-zA-z0-9 ]", "").toLowerCase().split("\\s+");
                System.out.println(WordVec.length);

                int[] InputVec = new int[1000];
                // Initialize the input
                for (int i = 0; i < 1000; i++) {
                    InputVec[i] = 0;
                }
                // Convert the words by their indices

                int i = 1000 - 1 ;
                for (int k = WordVec.length -1 ; k > -1 ; k--) {
                    try {
                        InputVec[i] = WordToInd.get(WordVec[k]);
                        System.out.println(WordVec[k]);
                        System.out.println(InputVec[i]);

                    }
                    catch (Exception e) {
                        InputVec[i] = 0;

                    }
                    i = i-1;
                }

                if (mInferenceInterface == null) {
                    AssetManager assetManager = getAssets();
                    mInferenceInterface = new TensorFlowInferenceInterface(assetManager,MODEL_FILE);
                }


                float[] value_ = new float[1];


                mInferenceInterface.feed(INPUT_NODE,InputVec,1,1000);
                mInferenceInterface.run(new String[] {OUTPUT_NODE}, false);
                System.out.println(Float.toString(value_[0]));
                mInferenceInterface.fetch(OUTPUT_NODE, value_);
                System.out.println(Float.toString(value_[0]));

                //System.out.println(reviewInput);
                double scoreIn;
                scoreIn = value_[0]*5;
                double ratingIn = scoreIn;
                String stringDouble = Double.toString(scoreIn);
                score.setText(stringDouble);
                ratingBar.setRating((float) ratingIn);


            }

        });

    }
}














