# Move to data folder
cd data

# Write the name of the .zip you are going to extract
SOURCEFOLDERNAME="0_raw"
TARGETFOLDERNAME="2_generated"
NAME="2step_transformation_dt4h_GPT4omini"
TARGET=$TARGETFOLDERNAME/$NAME
LANGUAGE1="en"
LANGUAGE2="nl"

# Create destination folder
echo "Creating folder $FOLDERNAME"
mkdir -p $TARGET

# Unzip the file and move it to the destination folder if exists
echo "Unzipping $FOLDERNAME.zip"
ZIPFILE="$SOURCEFOLDERNAME/$NAME.zip"
if [ -f "$ZIPFILE" ]; then
    unzip "$ZIPFILE" -d $TARGET
    # rm "$FOLDERNAME.zip"
fi

# For 2 steps translation
echo "Creating folders $LANGUAGE1 and $LANGUAGE2"
mkdir -p $TARGET/$LANGUAGE1
mkdir -p $TARGET/$LANGUAGE2

# Move the files to the corresponding folders
echo "Moving files to $LANGUAGE1 and $LANGUAGE2 if exists"
mv $TARGET/*_transformed_step1.txt $TARGET/$LANGUAGE1/
mv $TARGET/*_transformed_step2.txt $TARGET/$LANGUAGE2/

echo "Done"