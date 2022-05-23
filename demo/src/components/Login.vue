<template>
  <div style="margin-left: 35%;margin-top: 150px;">
  <div style="width: 500px;height: 500px;">
    <div style="width: 400px;font-size: 25px;margin-left:70px;margin-bottom: 50px;font-weight: bold;color: #2F93FE">人脸颜值打分   DEMO</div>
  <el-upload
      drag
      action="https://jsonplaceholder.typicode.com/posts/"
      :show-file-list="false"
      :on-success="handleAvatarSuccess"
      :before-upload="beforeAvatarUpload">
    <img v-if="imageUrl" :src="imageUrl">
    <div v-else class="el-upload__text" style="margin-top: 20%">将图片拖到此处，或<em>点击上传</em></div>
  </el-upload>
  <el-rate
      v-model="value"
      disabled
      show-score
      text-color="#ff9900"
      score-template="颜值得分：{value} 颗星"
  />
  </div>
  </div>
</template>

<script>
import request from "../utils/request";
export default {
  name: "Login",
  data() {
    return{
      value:0,
      imageUrl: '',
    }
  },
  methods:{
    handleAvatarSuccess(res, file) {
      this.imageUrl = URL.createObjectURL(file.raw);
      this.$message.success('图片上传成功');
      var formData = new FormData();
      formData.append("img", file.raw);
      request.post("/api/pred",formData).then(res =>{
        this.value=parseInt(res);
      })
    },
    beforeAvatarUpload(file) {
      const isJPG = file.type === 'image/jpeg';
      const isLt2M = file.size / 1024 / 1024 < 2;

      if (!isJPG) {
        this.$message.error('上传图片只能是 JPG 格式!');
      }
      if (!isLt2M) {
        this.$message.error('上传图片大小不能超过 2MB!');
      }
      return isJPG && isLt2M;
    }
},

}
</script>
<style scoped>
</style>


