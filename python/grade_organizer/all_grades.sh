python summarize_grades.py data/qmF17.csv > names.txt
python summarize_grades.py data/qmF17.csv --dump-grades | grep 'Final exam' | grep ':' > final_exam.txt
python summarize_grades.py data/qmF17.csv --dump-grades | grep 'Final grade' > final_grade.txt


#paste names.txt final_exam.txt final_grade.txt

paste names.txt hw.txt midterm.txt final_exam.txt final_grade.txt
