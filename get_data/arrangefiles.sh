for i in {001..030}
do
   mkdir "$1$i"
   mv *"${1}${i}"*.nc "$1$i"
done
