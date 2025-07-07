import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from pathlib import Path
import argparse

class WaveformVisualizer:
    def __init__(self, folder_path, output_dir=None):
        """
        初始化声波可视化器
        
        Args:
            folder_path (str): 包含.wav文件的文件夹路径
            output_dir (str): 保存图像的输出目录，如果为None则不保存
        """
        self.folder_path = Path(folder_path)
        self.output_dir = Path(output_dir) if output_dir else None
        self.wav_files = self._get_wav_files()
        
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_wav_files(self):
        """获取文件夹中所有的.wav文件"""
        if not self.folder_path.exists():
            raise FileNotFoundError(f"文件夹不存在: {self.folder_path}")
        
        wav_files = list(self.folder_path.glob("*.wav"))
        print(f"找到 {len(wav_files)} 个.wav文件")
        return wav_files
    
    def load_audio(self, file_path):
        """
        加载音频文件
        
        Args:
            file_path (Path): 音频文件路径
            
        Returns:
            tuple: (音频数据, 采样率)
        """
        try:
            # 使用librosa加载音频，自动转换为单声道
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            return audio_data, sample_rate
        except Exception as e:
            print(f"加载音频文件失败 {file_path}: {e}")
            return None, None
    
    def plot_waveform(self, audio_data, sample_rate, title, save_path=None):
        """
        绘制波形图
        
        Args:
            audio_data (np.array): 音频数据
            sample_rate (int): 采样率
            title (str): 图表标题
            save_path (Path): 保存路径，如果为None则不保存
        """
        # 创建时间轴
        duration = len(audio_data) / sample_rate
        time = np.linspace(0, duration, len(audio_data))
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 绘制波形
        plt.subplot(2, 1, 1)
        plt.plot(time, audio_data, linewidth=0.5, color='blue')
        plt.title(f'{title} - 波形图')
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅度')
        plt.grid(True, alpha=0.3)
        
        # 绘制频谱图
        plt.subplot(2, 1, 2)
        # 计算短时傅里叶变换
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        # 转换为分贝
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # 绘制频谱图
        librosa.display.specshow(magnitude_db, sr=sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'{title} - 频谱图')
        plt.xlabel('时间 (秒)')
        plt.ylabel('频率 (Hz)')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
    
    def plot_multiple_waveforms(self, max_files=None):
        """
        在同一个图中绘制多个波形（用于比较）
        
        Args:
            max_files (int): 最大显示文件数量，None表示显示所有
        """
        if not self.wav_files:
            print("没有找到.wav文件")
            return
        
        files_to_plot = self.wav_files[:max_files] if max_files else self.wav_files
        
        plt.figure(figsize=(15, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(files_to_plot)))
        
        for i, (wav_file, color) in enumerate(zip(files_to_plot, colors)):
            audio_data, sample_rate = self.load_audio(wav_file)
            if audio_data is not None:
                # 归一化时间轴
                duration = len(audio_data) / sample_rate
                time = np.linspace(0, duration, len(audio_data))
                
                # 为了清晰显示，给每个波形添加偏移
                offset = i * 0.5
                plt.plot(time, audio_data + offset, 
                        label=wav_file.name, color=color, linewidth=0.7)
        
        plt.title('多个音频文件波形对比')
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅度 (带偏移)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.output_dir:
            save_path = self.output_dir / "multiple_waveforms_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存到: {save_path}")
        
        plt.show()
    
    def analyze_all_files(self):
        """分析文件夹中的所有.wav文件"""
        if not self.wav_files:
            print("没有找到.wav文件")
            return
        
        print(f"开始分析 {len(self.wav_files)} 个音频文件...")
        
        for i, wav_file in enumerate(self.wav_files, 1):
            print(f"\n处理文件 {i}/{len(self.wav_files)}: {wav_file.name}")
            
            # 加载音频
            audio_data, sample_rate = self.load_audio(wav_file)
            if audio_data is None:
                continue
            
            # 打印基本信息
            duration = len(audio_data) / sample_rate
            print(f"  - 采样率: {sample_rate} Hz")
            print(f"  - 时长: {duration:.2f} 秒")
            print(f"  - 样本数: {len(audio_data)}")
            print(f"  - 最大幅度: {np.max(np.abs(audio_data)):.4f}")
            
            # 绘制波形
            save_path = None
            if self.output_dir:
                save_path = self.output_dir / f"{wav_file.stem}_waveform.png"
            
            self.plot_waveform(audio_data, sample_rate, wav_file.stem, save_path)
    
    def get_audio_statistics(self):
        """获取所有音频文件的统计信息"""
        if not self.wav_files:
            print("没有找到.wav文件")
            return
        
        stats = []
        for wav_file in self.wav_files:
            audio_data, sample_rate = self.load_audio(wav_file)
            if audio_data is not None:
                duration = len(audio_data) / sample_rate
                max_amplitude = np.max(np.abs(audio_data))
                rms = np.sqrt(np.mean(audio_data**2))
                
                stats.append({
                    'filename': wav_file.name,
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'samples': len(audio_data),
                    'max_amplitude': max_amplitude,
                    'rms': rms
                })
        
        # 打印统计表格
        print("\n音频文件统计信息:")
        print("-" * 80)
        print(f"{'文件名':<25} {'时长(s)':<8} {'采样率':<8} {'最大幅度':<10} {'RMS':<10}")
        print("-" * 80)
        
        for stat in stats:
            print(f"{stat['filename']:<25} {stat['duration']:<8.2f} "
                  f"{stat['sample_rate']:<8} {stat['max_amplitude']:<10.4f} "
                  f"{stat['rms']:<10.4f}")
        
        return stats

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='音频波形可视化工具')
    parser.add_argument('folder_path', help='包含.wav文件的文件夹路径')
    parser.add_argument('--output', '-o', help='输出图像的目录')
    parser.add_argument('--compare', '-c', action='store_true', 
                       help='生成多文件对比图')
    parser.add_argument('--max-compare', type=int, default=5,
                       help='对比图中最大文件数量 (默认: 5)')
    parser.add_argument('--stats-only', action='store_true',
                       help='只显示统计信息，不生成图表')
    
    args = parser.parse_args()
    
    # 创建可视化器
    try:
        visualizer = WaveformVisualizer(args.folder_path, args.output)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 显示统计信息
    stats = visualizer.get_audio_statistics()
    
    if args.stats_only:
        return
    
    if args.compare:
        # 生成对比图
        visualizer.plot_multiple_waveforms(args.max_compare)
    else:
        # 分析所有文件
        visualizer.analyze_all_files()

# 示例用法
if __name__ == "__main__":
    # 如果直接运行脚本，使用示例路径
    import sys
    
    if len(sys.argv) == 1:
        # 示例用法
        print("音频波形可视化工具")
        print("用法示例:")
        print("  python acousticWaveVisualization.py /path/to/wav/files")
        print("  python acousticWaveVisualization.py /path/to/wav/files --output ./output")
        print("  python acousticWaveVisualization.py /path/to/wav/files --compare")
        print("  python acousticWaveVisualization.py /path/to/wav/files --stats-only")
        
        # 交互式输入
        folder_path = input("\n请输入包含.wav文件的文件夹路径: ").strip()
        if folder_path and os.path.exists(folder_path):
            visualizer = WaveformVisualizer(folder_path)
            choice = input("选择操作: (1)分析所有文件 (2)生成对比图 (3)仅统计信息 [1]: ").strip()
            
            if choice == "3":
                visualizer.get_audio_statistics()
            elif choice == "2":
                visualizer.plot_multiple_waveforms(5)
            else:
                visualizer.analyze_all_files()
        else:
            print("无效的文件夹路径")
    else:
        main()