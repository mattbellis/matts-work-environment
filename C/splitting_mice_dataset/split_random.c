#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char **argv)
{
    char *infilename = argv[1];
    printf("%s\n",infilename);

    char outfilename[256];
    sprintf(outfilename,"random_file_%04d.dat",0);

    FILE *ifp = fopen(infilename, "r");
    FILE *ofp = NULL;
    float ra,dec,z;
    float z_v,x_c,y_c,z_c;
    int id;
    char dummy[256];

    // Seed the random number generator with the time.
    srand(time(NULL));

    if (ifp == NULL) {
        printf("Can't open input file %s!",infilename);
        exit(1);
    }

    printf("Opening %s\n",outfilename);
    ofp = fopen(outfilename, "w");
    if (ofp == NULL) {
        fprintf(stderr, "Can't open output file %s!\n", "out.dat");
        exit(1);
    }

    int count = 0;
    int fill_count = 0;
    //float probability_of_accepting_an_entry = 0.005;
    // for 689
    int tot_entries = 25984717;
    //int num_entries_per_file = 10000 + 500; // Kludge it to make sure we get enough galaxies.
    //int num_entries_per_file = 1000 + 50; // Kludge it to make sure we get enough galaxies.
    //int num_entries_per_file = 5000 + 200; // Kludge it to make sure we get enough galaxies.
    //int num_entries_per_file = 50000 + 100; // Kludge it to make sure we get enough galaxies.
    //int num_entries_per_file = 100000 + 200; // Kludge it to make sure we get enough galaxies.
    int num_entries_per_file = 500000 + 1000; // Kludge it to make sure we get enough galaxies.
    float probability_of_accepting_an_entry = (float)num_entries_per_file/tot_entries;
    float r;
    printf("%f",probability_of_accepting_an_entry);
    //exit(-1);

    fscanf(ifp, "%s,%s,%s,%s,%s,%s,%s",&dummy,&dummy,&dummy,&dummy,&dummy,&dummy,&dummy);

    //while (fscanf(ifp, "%f,%f,%f",&ra,&dec,&z) != EOF) {
    //while (fscanf(ifp, "%f\t%f\t%f\t%f\t%f\t%f\t%f",&ra,&dec,&z,&z_v,&x_c,&y_c,&z_c) != EOF) 
    // For 689
    //id,ra,dec,z,x_c,y_c,z_c
    while (fscanf(ifp, "%d,%f,%f,%f,%f,%f,%f",&id,&ra,&dec,&z,&x_c,&y_c,&z_c) != EOF) 
    {
        //printf("%f %f %f\n",ra,dec,z);
        //fprintf(ofp,"%f %f %f\n",ra,dec,z);
        r = (float)rand()/(float)RAND_MAX;
        //printf("%f\n",r);
        //exit(1);
        if (r<probability_of_accepting_an_entry)
        {
            //fprintf(ofp,"%f %f %f %f %f %f %f\n",ra,dec,z,z_v,x_c,y_c,z_c);
            // For 689
            fprintf(ofp,"%d %f %f %f %f %f %f\n",id,ra,dec,z,x_c,y_c,z_c);
            fill_count += 1;
        }

        if (count%100000==0)
            printf("%d\n",count);

        if (fill_count%10000==0 && fill_count!=0)
            printf("fill_count: %d\n",fill_count);

        count += 1;
    }

    fclose(ifp);
    fclose(ofp);

    return 0;
}
