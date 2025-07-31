# custom libraries
from feature.sift import SIFTMatcher
from feature.ransac import RANSAC
from stitching.homography import Homography, final_size
from stitching.apap import Apap
from stitching.blend import uniform_blend
from utils.mesh import get_mesh, get_vertice
from utils.draw import *
from utils.recursive import RecursiveDivider

# basic libraries
import time
import os
from PIL import Image
import glob
import re
from loguru import logger
import skimage
import cv2
import csv

"""
***  Conventional Image-Stitching Pipeline ***
1. image load
2. grayscaling & SIFT
3. Brute-force MATCHING
4. RANSAC
5. Estimate Global-Homography & extract Final size
6. Estimate Local-Homograhpy
7. Warping
8. Blending
"""


class ThreadSVComp:
    def __init__(self, opt):
        self.opt = opt
        self.n = 0  # line number of txt file
        self.unit_w, self.unit_h = self.opt.resize
        self.psnr_list = []
        self.ssim_list = []
        self.csv_path = os.path.join(self.opt.imgroot, f'apap_{self.opt.imgnum}.csv')
        self.already_processed = []

    def recursive(self, imgdir):
        if isinstance(imgdir, list):
            if len(imgdir) == 2:
                return self.thread(self.recursive(imgdir[0]), self.recursive(imgdir[1]))
            else:
                return self.recursive(imgdir[0])
        else:
            src = cv2.imread(imgdir[0], cv2.IMREAD_COLOR)
            assert src is not None, print(f'No such directory exists:{imgdir[0]}')
            src = src[:, :, ::-1]
            src = cv2.resize(src, dsize=(self.unit_w, self.unit_h))
            try:
                # process stitching
                dst = cv2.imread(imgdir[1], cv2.IMREAD_COLOR)[:, :, ::-1]
                dst = cv2.resize(dst, dsize=(self.unit_w, self.unit_h))
                return self.thread(src, dst)
            except:
                # just return
                return src

    def process(self, imgdir, mask, save_dir):
        unit_start = time.perf_counter()
        if self.opt.print_n: print(f'processing {self.n} thread...')
        # ========================================== call image & stitch ==============================================
        result = self.recursive(imgdir)
        # ==============================================================================================================
        # ====================================== mask carving on result image ==========================================
        if isinstance(mask, str):
            mask_img = cv2.imread(mask, cv2.IMREAD_COLOR)[:, :, ::-1]
            mask_img = cv2.normalize(mask_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            h, w, _ = mask_img.shape
            result = cv2.resize(result, dsize=(w, h))
            result *= mask_img
        elif isinstance(mask, list):
            mask_img = cv2.imread(mask[self.n], cv2.IMREAD_COLOR)[:, :, ::-1]
            mask_img = cv2.normalize(mask_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            h, w, _ = mask_img.shape
            result = cv2.resize(result, dsize=(w, h))
            result *= mask_img
        else:
            pass
        # ==============================================================================================================
        # =========================================== save result image ================================================
        if self.opt.save:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, str(self.opt.imgnum)+'.'+self.opt.savefmt)
            cv2.imwrite(save_path, result[:, :, ::-1])
            if self.opt.saveprint: print(f'{self.n} image saved -> {save_path}')
        # ==============================================================================================================
        # =========================================== print result image ===============================================
        if self.opt.image_show > self.n or self.opt.image_show == -1:
            result = Image.fromarray(result)
            result.show()
        # ==============================================================================================================
        if self.opt.unit_time: print(f'{self.n} image time spending: {time.perf_counter() - unit_start:4f}s.')

    @staticmethod
    def call_dataset(fname, root=None):
        file = open(fname, 'r')
        data = file.readlines()
        target_stack = []
        for d in data:
            imgname = d.strip().strip('\n').split(' ')
            if root is not None:
                # path merging
                target_stack.append([os.path.join(root, name) for name in imgname])
            else:
                # absolute path
                target_stack.append(imgname)
        return target_stack

    def call_mask(self):
        if self.opt.mask_dir is None:
            return None
        try:  # only one mask
            mask = self.opt.mask_dir
        except:  # mask text file
            mask = self.call_dataset(self.opt.mask_dir, root=self.opt.mask_root)
        return mask

    def thread(self, src, dst):
        mesh_size = self.opt.mesh_size

        img1 = src
        img2 = dst
        ori_h, ori_w, _ = img1.shape
        dst_h, dst_w, _ = img2.shape
        sift = SIFTMatcher()

        if self.opt.verbose: print(f'{self.n} image SIFT...')
        # SIFT & KNN BFMatching
        src_match, dst_match = sift.thread(img1, img2)
        if self.opt.verbose: print(f"raw matching points: {len(src_match)}")

        # RANSAC
        if self.opt.verbose: print(f'{self.n} image RANSAC...')
        ransac = RANSAC(self.opt)
        final_src, final_dst = ransac.thread(src_match, dst_match, self.opt.ransac_max)
        if self.opt.verbose: print(f'final matching points: {len(final_src)}')

        # 检查RANSAC是否返回了有效的匹配点
        if len(final_src) == 0:
            logger.warning(f"RANSAC failed - no valid matching points found")
            # 创建一个简单的拼接结果
            final_h = max(ori_h, dst_h)
            final_w = max(ori_w, dst_w)
            result = np.zeros(shape=(final_h, final_w, 3), dtype=np.uint8)
            warped_img = cv2.resize(img1, dsize=(final_w, final_h))
            dst_img = cv2.resize(img2, dsize=(final_w, final_h))
            result = uniform_blend(warped_img, dst_img)  # 直接叠加，相当于单位阵拼接
            # result[:, :int(final_w/2), :] = img1[:, :int(final_w/2), :]
            # result[:, int(final_w/2):, :] = img2[:, int(final_w/2):, :]
            # 计算拼接指标
            if self.opt.metric:
                self.metric(warped_img, dst_img, result)
            return result

        # Global Homography
        if self.opt.verbose: print(f'{self.n} image Global Homography Estimation...')
        h_agent = Homography()
        gh = h_agent.global_homography(final_src, final_dst)
        final_w, final_h, offset_x, offset_y = final_size(img1, img2, gh)

        if abs(final_h) > ori_h * 4. or abs(final_w) > ori_w * 4.:  # 检查拼接后的图像尺寸是否超过原始图像尺寸的4倍
            logger.warning("Homography Estimation Failed !")
            final_h = max(ori_h, dst_h)
            final_w = max(ori_w, dst_w)
            result = np.zeros(shape=(final_h, final_w, 3), dtype=np.uint8)
            warped_img = cv2.resize(img1, dsize=(final_w, final_h))
            dst_img = cv2.resize(img2, dsize=(final_w, final_h))
            result = uniform_blend(warped_img, dst_img)  # 直接叠加，相当于单位阵拼接
            # result[:, :int(final_w/2), :] = img1[:, :int(final_w/2), :]
            # result[:, int(final_w/2):, :] = img2[:, int(final_w/2):, :]
        else:
            # APAP
            # ready meshgrid
            mesh = get_mesh((final_w, final_h), mesh_size + 1)
            vertices = get_vertice((final_w, final_h), mesh_size, (offset_x, offset_y))

            # As-Projective-As-Possible Stitcher instance definition
            stitcher = Apap(self.opt, [final_w, final_h], [offset_x, offset_y])
            # local homography estimating
            if self.opt.verbose: print(f'{self.n} image local homography Estimation...')
            local_homography, local_weight = stitcher.local_homography(final_src, final_dst, vertices)
            # local warping
            if self.opt.verbose: print(f'{self.n} image local warping...')
            warped_img = stitcher.local_warp(img1, local_homography, mesh, self.opt.warping_progress)

            # another image pixel move
            dst_img = np.zeros_like(warped_img)
            dst_img[offset_y: dst_h + offset_y, offset_x: dst_w + offset_x, :] = img2

            # Uniform(50:50) blending
            if self.opt.verbose: print(f'{self.n} image blending...')
            result = uniform_blend(warped_img, dst_img)

            # Draw
            if self.opt.match_print:
                match_fig = draw_match(img1, img2, final_src, final_dst, self.opt.matching_line)
                Image.fromarray(match_fig).show()

        # 计算拼接指标
        if self.opt.metric:
            self.metric(warped_img, dst_img, result)

        return result

    @staticmethod
    def call_dataset_sv_comp(root, imgnum):      
        target_stack = []
        groups = glob.glob(os.path.join(root, 'testing', '*'))  # 当前子数据集中的所有数据组
        pattern = r'^\d{4,5}$'  # 匹配仅由四位或五位数字组成的字符串
        for group in sorted(groups):
            group_idx = group.split('/')[-1]
            if bool(re.match(pattern, group_idx)):  # 如果当前数据组的名称符合要求
                img_lists = glob.glob(os.path.join(group, '*.jpg'))
                img_lists.sort()
                if len(img_lists) < imgnum:
                    continue
                target_stack.append(img_lists[:imgnum])

        return target_stack

    def metric(self, warped_img, dst_img, result):
        # 生成掩码
        warped_mask = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.uint8)
        warped_mask[np.any(warped_img > 0, axis=2)] = 255
        dst_mask = np.zeros((dst_img.shape[0], dst_img.shape[1]), dtype=np.uint8)
        dst_mask[np.any(dst_img > 0, axis=2)] = 255

        # 掩码中白色区域可能存在细小黑点，通过形态学操作去除
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 形态学操作去除细小黑点
        warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, kernel)  # 闭操作：先膨胀后腐蚀，填充小的黑点（在白色区域中）
        dst_mask = cv2.morphologyEx(dst_mask, cv2.MORPH_CLOSE, kernel)
        warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_OPEN, kernel)  # 开操作：先腐蚀后膨胀，去除小的白点（在黑色背景上）
        dst_mask = cv2.morphologyEx(dst_mask, cv2.MORPH_OPEN, kernel)

        # 计算重叠区域指标
        warped_mask_f = cv2.cvtColor(warped_mask.astype(np.float32)/255, cv2.COLOR_GRAY2BGR)
        dst_mask_f = cv2.cvtColor(dst_mask.astype(np.float32)/255, cv2.COLOR_GRAY2BGR)
        overlap_mask = warped_mask_f * dst_mask_f
        psnr_one = skimage.measure.compare_psnr(warped_img*overlap_mask, dst_img*overlap_mask, 255)
        ssim_one = skimage.measure.compare_ssim(warped_img*overlap_mask, dst_img*overlap_mask, data_range=255, multichannel=True)
        logger.info(f'psnr: {psnr_one}, ssim: {ssim_one}')
        self.psnr_list.append(psnr_one)
        self.ssim_list.append(ssim_one)
        self.csv_writer.writerow([self.n, psnr_one, ssim_one])

        # 保存调试图像
        # cv2.imwrite('warped_img.jpg', warped_img)
        # cv2.imwrite('dst_img.jpg', dst_img)
        # cv2.imwrite('result.jpg', result)
        # cv2.imwrite('warped_mask.jpg', warped_mask)
        # cv2.imwrite('dst_mask.jpg', dst_mask)
        # cv2.imwrite('overlap_mask.jpg', overlap_mask.astype(np.uint8)*255)
    
    def forward(self):
        # mask setting
        mask = self.call_mask()
        # divider instance
        divider = RecursiveDivider()
        # 加载数据
        datalist = self.call_dataset_sv_comp(self.opt.imgroot, self.opt.imgnum)

        # 从 csv 文件中恢复测试数据
        if os.path.exists(self.csv_path):
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self.already_processed.append(int(row[0]))
                    self.psnr_list.append(float(row[1]))
                    self.ssim_list.append(float(row[2]))
        
        with open(self.csv_path, 'a', newline='') as f:
            self.csv_writer = csv.writer(f)  # 追加写入
            # 进行拼接
            for idx in range(len(datalist)):
                if idx in self.already_processed:  # 跳过已经处理过的数据
                    continue
                logger.info(f'====================== {idx} / {len(datalist)} ======================')
                self.n = idx
                data = datalist[idx]
                save_dir = os.path.join(os.path.dirname(data[0]), 'apap')
                data = divider.list_divide(data)
                self.process(data, mask, save_dir)
            # 关闭CSV文件
            f.close()
            
        # 计算指标
        logger.info('<==================== Analysis ===================>')
        total_samples = len(datalist)
        failure_num = 0  # TODO: 计算失败数量
        thirty_percent_index = int((total_samples-failure_num) * 0.3)
        sixty_percent_index = int((total_samples-failure_num) * 0.6)
        logger.info(f'Fail num: {failure_num}, total num: {total_samples}, success num: {total_samples-failure_num}')
        
        self.psnr_list.sort(reverse = True)
        psnr_list_30 = self.psnr_list[0 : thirty_percent_index]
        psnr_list_60 = self.psnr_list[thirty_percent_index: sixty_percent_index]
        psnr_list_100 = self.psnr_list[sixty_percent_index: -1]
        print("[psnr] top 30%: ", np.mean(psnr_list_30))
        print("[psnr] top 30~60%: ", np.mean(psnr_list_60))
        print("[psnr] top 60~100%: ", np.mean(psnr_list_100))
        logger.info('[psnr] average: {}'.format(np.mean(self.psnr_list)))

        self.ssim_list.sort(reverse = True)
        ssim_list_30 = self.ssim_list[0 : thirty_percent_index]
        ssim_list_60 = self.ssim_list[thirty_percent_index: sixty_percent_index]
        ssim_list_100 = self.ssim_list[sixty_percent_index: -1]
        print("[ssim] top 30%: ", np.mean(ssim_list_30))
        print("[ssim] top 30~60%: ", np.mean(ssim_list_60))
        print("[ssim] top 60~100%: ", np.mean(ssim_list_100))
        logger.info('[ssim] average: {}'.format(np.mean(self.ssim_list)))        
