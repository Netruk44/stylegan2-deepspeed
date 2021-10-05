# stylegan2-deepspeed

An implementation of stylegan2 using Microsoft's DeepSpeed.

Currently WIP, lots of things unimplemented.

GAN implemenation lifted from [lucidrain's](https://github.com/lucidrains/stylegan2-pytorch) repo.

# Example Commandline
``` shell
deepspeed `which stylegan2_deepspeed` --deepspeed --deepspeed_config ./config/stylegan2.json -d /path/to/input/images
```

# References

TODO