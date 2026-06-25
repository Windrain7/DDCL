function compute_metrics(gt_folder, folder)
    gt_img_files = [dir(fullfile(gt_folder, '*.png')); dir(fullfile(gt_folder, '*.jpg'))];
    fprintf('length of gt_img_files: %d\n', length(gt_img_files));
    output_filename = fullfile(folder, '0_psnr_ssim.log');

    % 初始化 PSNR 和 SSIM 数组
    nimgs = length(gt_img_files);
    psnrs = zeros(nimgs, 1);
    ssims = zeros(nimgs, 1);
    
    % 新增：初始化一个单元格数组来存储每张图片的日志行
    img_logs = cell(nimgs, 1);

    % 遍历图像文件
    for i = 1:nimgs
        % 获取原始图像和对应的 _gt.jpg 图像的文件名
        gt_img_name = gt_img_files(i).name;
        img_name = gt_img_name;
        if ~isfile(fullfile(folder, img_name))
            [~, name, ~] = fileparts(img_name);
            img_name = [name, '.jpg'];
        end
        
        if ~isfile(fullfile(folder, img_name))
            [~, name, ~] = fileparts(img_name);
            img_name = [name, '.png'];
        end

        % 读取图像
        x_gt = im2double(imread(fullfile(gt_folder, gt_img_name)));
        x = im2double(imread(fullfile(folder, img_name)));
        
        % 转换为 Y 通道
        x_gt = rgb2ycbcr(x_gt); x_gt = x_gt(:,:,1);
        x = rgb2ycbcr(x); x = x(:,:,1);
        
        % 计算 PSNR 和 SSIM
        ts = ssim(x_gt * 255, x * 255);
        tp = psnr(x_gt, x);
        
        % 输出结果到控制台
        fprintf('%s, PSNR: %6.4f, SSIM: %6.4f\n', img_name, tp, ts);
        
        % 存储结果
        psnrs(i) = tp;
        ssims(i) = ts;

        % 新增：将单张图片的日志行存储在内存中
        img_logs{i} = sprintf('%s, PSNR: %6.4f, SSIM: %6.4f\n', img_name, tp, ts);
    end

    % 计算平均 PSNR 和 SSIM
    mean_psnr = mean(psnrs);
    mean_ssim = mean(ssims);
    
    % 输出平均结果到控制台
    fprintf('Average PSNR/SSIM: %6.2f/%6.3f\n', mean_psnr, mean_ssim);

    % ------ 写入文件的部分开始 ------
    % 打开日志文件，'w' 模式会清空文件内容
    fid = fopen(output_filename, 'w');

    % 1. 首先写入平均值
    fprintf(fid, 'Average PSNR/SSIM: %6.2f/%6.3f\n', mean_psnr, mean_ssim);
    
    % 2. 然后遍历之前存储的单元格数组，写入每张图片的详细数据
    for i = 1:nimgs
        fprintf(fid, '%s', img_logs{i});
    end

    % 3. 关闭日志文件
    fclose(fid);
end