output=$1
if [ ! -d $output ]; then
  echo "$output does not exist"
  echo "Create output folder: $output"
  mkdir -p $output/IMG
fi
shift
files=""
for folder in $@
do
  if [ -d $folder ]; then
    echo $folder
    files="$files $folder/driving_log.csv" 
    ls -l $folder/IMG | wc -l
    find $folder/IMG -name '*.jpg' -exec cp {} $output/IMG \;
  fi
done

echo "files to be combined are $files" 
echo "files are combined to $output/driving_log.csv"
cat $files > $output/driving_log.csv
