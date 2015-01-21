      subroutine MPI_INIT(ierr)
        write(*,*) ' MPI-stub used!'
      return
      end
      subroutine MPI_FINALIZE(ierr)
      return
      end
      subroutine MPI_INITIALIZED(flag,ierr)
      logical flag
      flag=.false.
      return
      end
      subroutine MPI_COMM_SIZE(mpicomm,mpisize,ierr)
      mpisize=1
      return
      end
      subroutine MPI_COMM_RANK(mpicomm,mpirank,ierr)
      mpirank=0
      return
      end
      real*8 Function MPI_WTIME()
	use dfport
	real*4 t(2)
	call etime(t)
      mpi_wtime=1.0d0*t(1)
      return
      end 
      subroutine MPI_BARRIER(mcomm,ierr)
      return
      end
      subroutine MPI_BCAST(b,n,nt,nr,mcomm,ierr)
      return
      end
      subroutine MPI_SEND(b,n,nt,mt,nr,mcomm,ierr)
      return
      end
      subroutine MPI_RECV(b,n,nt,mt,nr,mcomm,ms,ierr)
      return
      end
      subroutine MPI_PROBE(nr,mt,mcomm,ms,ierr)
      return
      end
      subroutine MPI_GET_COUNT(ms,nt,n,ierr)
      n=1
      return
      end
      subroutine MPI_COMM_DUP(mc1,mc2,ierr)
      return
      end
      subroutine MPI_COMM_FREE(mc,ierr)
      return
      end
      subroutine MPI_TYPE_COMMIT(mt,ierr)
      return
      end
      subroutine MPI_TYPE_FREE(mt,ierr)
      return
      end
      subroutine MPI_TYPE_VECTOR(ic,ib,is,mtold,mtnew,ierr)
      return
      end
