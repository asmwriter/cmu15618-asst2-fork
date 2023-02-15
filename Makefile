all:
	(cd saxpy ; make all)
	(cd scan ; make all)
	(cd render ; make all)

handin.tar: clean
	tar cvf handin.tar saxpy scan render Makefile 

clean:
	(cd saxpy ; make clean)
	(cd scan ; make clean)
	(cd render ; make clean)
	rm -f *~ handin.tar
