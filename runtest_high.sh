cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=release -B build
make -C build -j faiss
make -C build utils
make -C build test_bf_high
make -C build test_hnsw_high
make -C build test_acorn_high
make -C build test_cq_high

# now=$(date +"%m-%d-%Y")



efs=45

# dataset="sift"

for dataset in "sift" "gist"; do

    if [[ "$dataset" == "sift" ]]; then
        N=1000000
        gamma=30
        M=32
        M_beta=64
    elif [[ "$dataset" == "paper" ]]; then
        N=2029997
        gamma=30
        M=32
        M_beta=64
    elif [[ "$dataset" == "gist" ]]; then
        N=1000000
        gamma=30
        M=32
        M_beta=64
    fi

    # for((i=4;i<5;i++)) do
    #     ./build/demos/test_bf_high $N $gamma $dataset $M $M_beta $i
    # done

    export OMP_NUM_THREADS=64
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores

    for((i=4;i<6;i++)) do
        ./build/demos/test_hnsw_high $N $gamma $dataset $M $M_beta $i $efs
    done

    for((i=4;i<6;i++)) do
        ./build/demos/test_acorn_high $N $gamma $dataset $M $M_beta $i $efs
    done

    for((i=4;i<6;i++)) do
        ./build/demos/test_cq_high $N $gamma $dataset $M $M_beta $i $efs
    done
    
done


# run of sift1M test
# N=1000000 
# gamma=12 
# dataset=sift1M_test 
# M=32 
# M_beta=64
# parent_dir=${now}_${dataset}
# mkdir ${parent_dir}
# dir=${parent_dir}/MB${M_beta}
# mkdir ${dir}
# TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt
# ./build/demos/test_acorn $N $gamma $dataset $M $M_beta  &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt
