for tile_sizes in '200x200' '400x400' '200x200,400x400'
do
    for overlap in 0 15 30
    do
        echo "$tile_sizes $overlap"
        python3 detect-small.py \
            --input $HOME/videos/video1.mp4 \
            --output "$HOME/outputs/out_${tile_sizes}_${overlap}.mp4" \
            --tile_sizes $tile_sizes \
            --tile_overlap $overlap \
            --iou_threshold 0.1 \
            --score_threshold 0.5
    done
done