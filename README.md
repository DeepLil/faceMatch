1. Add Requirements file
2. Add setup file
3. Run setup file to install the requirements
3. create src folder
4. create a init file inside src folder
5. create a utils folder inside src folder
6. create a init file inside utils folder
7. create all_utils file inside utils folder
8. start writing all_utils file:
    write a function to read yaml file
    write a function to create directories 
9. create a folder config
10. write config file inside config folder
    create a folder structure of artifacts which contains
    where to store the pickle file of data
    where to store the pickle file of features extracted
11. create params file
    create a structure of base which contains
    data path
    base model
    transfer learning option (include top bool)
    pooling type
-------------------------------------------------------------------------------------------------------
12. create file 01_generate_pkl_img 
    parse through all folders and files in the data folder and read all the filenames and store them in a list and dump this list into a pickle file
    write with exception and log the exception
13. create file 02_feature_extractor 
    Import keras vggface
    Import keras vggface utils
    Load the datanames pickle file
    create a model object for vggface class
    parse through each filename and read the file and pass the file through the model object
    Output of the model is a finite length feature vector 
    Append all the feature vectors to a list
    Dump this list into a pickle file

14. Create app file
    Import opencv
    Import MTCNN
    Import cosine similarity metric 
    Create a detector object with MTCNN class
    Ask the user for file upload through streamlit
    Detect the face from the image through detector 
    Extract the face patch using opencv basic functions
    Pass this face through feature extractor from 13. 
    Compare the extracted feature to the feature extracted list using cosine similarity
    Sort the similairty 
    Output the image with highest similarity


