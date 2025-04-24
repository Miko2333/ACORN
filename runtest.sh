cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=release -B build
make -C build -j faiss
make -C build utils
make -C build test_bf
make -C build test_hnsw
make -C build test_acorn
make -C build test_cq

# now=$(date +"%m-%d-%Y")



efs=15

dataset="sift"

# for dataset in "sift" "gist" "paper"; do

    if [[ "$dataset" == "sift" ]]; then
        N=1000000
        gamma=12
        M=32
        M_beta=64
    elif [[ "$dataset" == "paper" ]]; then
        N=2029997
        gamma=12
        M=32
        M_beta=64
    elif [[ "$dataset" == "gist" ]]; then
        N=1000000
        gamma=12
        M=32
        M_beta=64
    fi

    # for((i=0;i<4;i++)) do
    #     ./build/demos/test_bf $N $gamma $dataset $M $M_beta $i
    # done

    export OMP_NUM_THREADS=64
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores

    for((i=0;i<4;i++)) do
        ./build/demos/test_hnsw $N $gamma $dataset $M $M_beta $i $efs
    done

    # for((i=0;i<4;i++)) do
    #     ./build/demos/test_acorn $N $gamma $dataset $M $M_beta $i $efs
    # done

    # for((i=0;i<4;i++)) do
    #     ./build/demos/test_cq $N $gamma $dataset $M $M_beta $i $efs
    # done
    
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
