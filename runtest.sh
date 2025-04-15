cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=release -B build
make -C build -j faiss
make -C build utils
make -C build test_acorn
make -C build test_bf
make -C build test_cq

now=$(date +"%m-%d-%Y")


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
./build/demos/test_acorn 1000000 12 sift1M 32 64 0
./build/demos/test_acorn 2029997 12 paper 32 64 0
./build/demos/test_bf 1000000 12 sift1M 32 64 0
./build/demos/test_bf 2029997 12 paper 32 64 0
./build/demos/test_cq 1000000 12 sift1M 32 64 0
./build/demos/test_cq 2029997 12 paper 32 64 0