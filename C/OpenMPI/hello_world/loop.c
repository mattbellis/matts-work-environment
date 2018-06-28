int main(int argc, char **argv)
{
    int a[1000000];

#pragma omp parallel for
    for (int i = 0; i < 1000000; i++) {
        a[i] = 2 * i;
    }

    return 0;
}
