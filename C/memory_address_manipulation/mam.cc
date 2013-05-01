#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

//int open(const char *pathname, int flags);
//int open(const char *pathname, int flags, mode_t mode);
//int creat(const char *pathname, mode_t mode);

int main()
{
    //open
    //write
    //close
    return 0;
}

ReadDrive(int MyByte)
{
    int fd;
    int result;
    unsigned char buf[512];

    fd = open("\\\\.\\PhysicalDrive1", 0);

    if (fd == -1)
    {
        return -2;
    }

    result = read(fd, buf, 512);

    close(fd);

    return buf[MyByte];
}

