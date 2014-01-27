rm nn_cut_options.txt >& /dev/null

plot_eff_rej_curves_from_text_files.py --ntp ntp1 --baryon LambdaC --batch >> nn_cut_options.txt
plot_eff_rej_curves_from_text_files.py --ntp ntp2 --baryon LambdaC --batch >> nn_cut_options.txt

plot_eff_rej_curves_from_text_files.py --ntp ntp1 --baryon Lambda0 --batch >> nn_cut_options.txt
plot_eff_rej_curves_from_text_files.py --ntp ntp2 --baryon Lambda0 --batch >> nn_cut_options.txt
plot_eff_rej_curves_from_text_files.py --ntp ntp3 --baryon Lambda0 --batch >> nn_cut_options.txt
plot_eff_rej_curves_from_text_files.py --ntp ntp4 --baryon Lambda0 --batch >> nn_cut_options.txt
