#echo "Exam"
#set date = '5/3/2012'
#python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep exam | grep $date | sort -k4 -n | awk '{if ($4>=90 && $4<200) print $4};' | wc -l
#python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep exam | grep $date | sort -k4 -n | awk '{if ($4>=80 && $4<90) print $4};' | wc -l
#python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep exam | grep $date | sort -k4 -n | awk '{if ($4>=70 && $4<80) print $4};' | wc -l
#python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep exam | grep $date | sort -k4 -n | awk '{if ($4>=60 && $4<70) print $4};' | wc -l
#python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep exam | grep $date | sort -k4 -n | awk '{if ($4>=0 && $4<60) print $4};' | wc -l
#
echo "Final exam"
python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep final_exam | sort -k4 -n | awk '{if ($4>=90 && $4<200) print $4};' | wc -l
python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep final_exam | sort -k4 -n | awk '{if ($4>=80 && $4<90) print $4};' | wc -l
python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep final_exam | sort -k4 -n | awk '{if ($4>=70 && $4<80) print $4};' | wc -l
python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep final_exam | sort -k4 -n | awk '{if ($4>=60 && $4<70) print $4};' | wc -l
python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep final_exam | sort -k4 -n | awk '{if ($4>=0 && $4<60) print $4};' | wc -l

echo "Final grade"
python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep 'Final grade' | sort -k3 -n | awk '{if ($3>=90 && $3<200) print $3};' | wc -l
python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep 'Final grade' | sort -k3 -n | awk '{if ($3>=80 && $3<90) print $3};' | wc -l
python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep 'Final grade' | sort -k3 -n | awk '{if ($3>=70 && $3<80) print $3};' | wc -l
python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep 'Final grade' | sort -k3 -n | awk '{if ($3>=60 && $3<70) print $3};' | wc -l
python summarize_grades.py data/NIU\ -\ PHYS\ 283\ -\ Spring\ 2012\ -\ Grades\ -\ Course\ grades.csv --dump-grades | grep 'Final grade' | sort -k3 -n | awk '{if ($3>=0 && $3<60) print $3};' | wc -l
