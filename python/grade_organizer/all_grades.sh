python summarize_grades.py data/csis200_F17.csv > names.txt
python summarize_grades.py data/csis200_F17.csv --dump-grades | grep 'HW' | grep 'avg:' > hw.txt
python summarize_grades.py data/csis200_F17.csv --dump-grades | grep 'Mid-term' | grep 'avg:' > midterm.txt
python summarize_grades.py data/csis200_F17.csv --dump-grades | grep 'Final project' | grep ':' > final_exam.txt
python summarize_grades.py data/csis200_F17.csv --dump-grades | grep 'Final grade' > final_grade.txt


#paste names.txt final_exam.txt final_grade.txt

paste names.txt hw.txt midterm.txt final_exam.txt final_grade.txt
