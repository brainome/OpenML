#! /bin/csh

if ( ! -d Data ) mkdir Data

foreach line ( `cat testfiles.csv` )
    set test = ( `echo $line | sed  's/,/ /g'` )
    # Skip lines with less than 5 parameters
    if ( $#test < 5 ) continue
    set Id     = $test[1]
    set Name   = $test[2]
    set Fname  = $test[3]
    set Target = $test[4]
    set Result = $test[5]
    # Skip first line
    if ( $Id == "Id" ) continue
    if ( $Id == "BAD" ) continue
    if ( $Id == "BIG" ) continue

    # If specified 1 arg match $1 to Name
    if ( $#argv == 1 && $1 != $Name ) continue

    # Check if data file is there. If not, download it
    if ( ! -e Data/$Fname ) then
        echo "### Downloading $Fname from openml.org"
        set csvlink = `wget -qO- https://www.openml.org/d/$Id | grep get_csv | sed 's/.*href="//g;s/">.*//g'`
        wget -q -O Data/$Fname $csvlink
    endif

    if ( ! -e Data/$Fname ) then
        echo "### ERROR - $Name data file $Fname was not downloaded"
        continue
    endif
    set fname  = `echo $Fname | sed 's/\..*//g' `
    set suffix = `echo $Fname | sed 's/.*\./\./g'`
    echo "############################################"
    echo "########## " $Name
    echo "############################################"
        python3 predictors/$fname.py Data/$fname.csv -validate
    endif
end
