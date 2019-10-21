import sys






if __name__=='__main__':

    try :
        setup_file = sys.argv[1]
    except :
        setup_file = 'setup'
        
    print(setup_file)