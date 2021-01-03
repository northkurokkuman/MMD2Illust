# -*- coding: utf-8 -*-
import os
import asyncio
import threading
import tempfile
import cv2
from PIL import Image, ImageOps, ImageEnhance
import moviepy.editor as editor
from kivy.app import App
from  kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.core.text import LabelBase, DEFAULT_FONT
import MMD2illust_util

####### for pyinstaller
import sys
from kivy.resources import resource_add_path, resource_find


UTF8 = 'utf-8'
AVI_EXTENTION = ".avi"
MP4_EXTENTION = ".mp4"
MP3_EXTENTION = ".mp3"
JPG_EXTENTION = ".jpg"
PNG_EXTENTION = ".png"
ILLUST_NAME_EXTENSION = "_illust"
EXPAND_NAME_EXTENSION = "_expanded"
TEMP_FILE_NAME = "TEMP"

Expand_Rate = 2.0
MMD2illustCheckboxID = "MMD2illustCheckbox"
OversizeCheckboxID = "OversizeCheckbox"
IllustExpandCheckboxID = "IllustExpandCheckbox"
ExpandRateSliderID = "ExpandRateSlider"
ExpandRateSliderTextID = "ExpandRateSliderText"
ProgressBarID = "ProgressBar"
ProgressBarLabelID = "ProgressBarLabel"

LabelBase.register(DEFAULT_FONT, "ipaexg.ttf")

is_MMD2illust = True
is_over_sized = True
expand_rate = Expand_Rate

class Main_Layout(RelativeLayout):
    def _open_settings(self):
        global is_MMD2illust
        global expand_rate
        layout = Sub_Layout()
        MMD2illust_checkbox = layout.ids[MMD2illustCheckboxID]
        oversize_checkbox = layout.ids[OversizeCheckboxID]
        illust_expand_checkbox = layout.ids[IllustExpandCheckboxID]
        expand_rate_checkbox_slider = layout.ids[ExpandRateSliderID]
        expand_rate_checkbox_slider_text = layout.ids[ExpandRateSliderTextID]
        MMD2illust_checkbox.active = is_MMD2illust
        oversize_checkbox.disabled = not is_MMD2illust
        illust_expand_checkbox.active = not is_MMD2illust
        expand_rate_checkbox_slider.disabled = is_MMD2illust
        oversize_checkbox.active = is_over_sized
        expand_rate_checkbox_slider_text.text = "--Expand Rate: " + str(expand_rate)
        expand_rate_checkbox_slider.value = int(expand_rate*10)
        app= App.get_running_app()
        app.set_layout(layout)

class Sub_Layout(Widget):
    def _return_to_main(self):
        layout = Main_Layout()
        app= App.get_running_app()
        app.set_layout(layout)
    def _on_expand_rate_change(self, value):
        global expand_rate
        expand_rate = value
        self.ids[ExpandRateSliderTextID].text = "--Expand Rate: " + str(expand_rate)
    def _on_process_type_checkbox_check(self, checkbox):
        global is_MMD2illust
        MMD2illust_checkbox = self.ids[MMD2illustCheckboxID]
        oversize_checkbox = self.ids[OversizeCheckboxID]
        illust_expand_checkbox = self.ids[IllustExpandCheckboxID]
        expand_rate_checkbox_slider = self.ids[ExpandRateSliderID]
        is_MMD2illust = (checkbox == MMD2illust_checkbox)
        MMD2illust_checkbox.active = is_MMD2illust
        oversize_checkbox.disabled = not is_MMD2illust
        illust_expand_checkbox.active = not is_MMD2illust
        expand_rate_checkbox_slider.disabled = is_MMD2illust
    def _on_is_oversize_checkbox_check(self, checkbox):
        global is_over_sized
        oversize_checkbox = self.ids[OversizeCheckboxID]
        is_over_sized = oversize_checkbox.active

class Process_Layout(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _cancel_process(self):
        layout = Main_Layout()
        app= App.get_running_app()
        app.process_thread.cancel_flag = True
        app.set_layout(layout)

    def set_progress_number(self, number, total_count = -1, num_processes=1, current_process=1):
        self.ids[ProgressBarID].max = total_count
        self.ids[ProgressBarLabelID].text = "Progress: " + str(number) + "/" + str(total_count) + \
            "(" + str(current_process) + "/" + str(num_processes) + ")"
        self.ids[ProgressBarID].value = number

    def set_progress_text(self, text):
        self.ids[ProgressBarLabelID].text = text

class MMD2illust_App(App):
    def build(self):
        from kivy.core.window import Window
        Window.size = (600, 400)
        Window.bind(on_dropfile = self._on_file_drop)
        Window.bind(on_resize = self._on_window_resize)
        self.layout = Main_Layout()
        return self.layout

    def set_layout(self, layout):
        Window.remove_widget(self.layout)
        self.layout = layout
        Window.add_widget(layout)

    async def mmd2illust_and_save(self, input_path, loop):
        if os.path.isdir(input_path):
            self.layout.set_progress_text("Processing...")
            image_fullnames = [filename for filename in os.listdir(input_path) if (JPG_EXTENTION in filename) | (PNG_EXTENTION in filename)]
            total_images_count = len(image_fullnames)
            processed_image_num = 0
            for filefullname in image_fullnames:
                thread = threading.currentThread()
                if getattr(thread, "cancel_flag", False):
                    loop.stop()
                    return

                input_image_path = input_path + "/" + filefullname
                name, ext = os.path.splitext(input_image_path)
                out = await self.process_image(input_image_path, loop)
                self.layout.set_progress_text("Saving...")
                output_path = name+ILLUST_NAME_EXTENSION+ext
                out.save(output_path)
                processed_image_num += 1
                self.layout.set_progress_number(processed_image_num , total_images_count)
        else:
            name, ext = os.path.splitext(input_path)
            if (ext == JPG_EXTENTION) | (ext == PNG_EXTENTION):
                self.layout.set_progress_text("Processing...")
                out = await self.process_image(input_path, loop)
                self.layout.set_progress_text("Saving...")
                output_path = name+ILLUST_NAME_EXTENSION+ext
                out.save(output_path)
            elif (ext == MP4_EXTENTION) | (ext == AVI_EXTENTION):
                with tempfile.TemporaryDirectory() as temp_folder_path:
                    print("video")
                    # temp_folder_path = os.path.dirname(input_path)
                    temp_video_path1 = temp_folder_path + "/" + TEMP_FILE_NAME + AVI_EXTENTION
                    real_size = (1024, 576)
                    capture, fps, original_width, original_height, total_frame_count, fourcc \
                        = MMD2illust_util.get_video_info(input_path)

                    ###### resizing video to (1024, 576)
                    from waifu2x_processor import waifu2x_processor
                    with waifu2x_processor() as processor:
                        def process1(frame):
                            nonlocal real_size
                            frame = MMD2illust_util.cv22pil(frame)
                            frame, is_pad, real_size = MMD2illust_util.pil_pad_to_size(frame)
                            while (not is_pad):
                                frame = processor.waifu2x_pil(frame)
                                frame, is_pad, real_size = MMD2illust_util.pil_pad_to_size(frame)
                            return frame
                        await self.process_video(input_path, temp_video_path1, loop, process1, 3, 1)

                    ###### converting video
                    temp_video_path2 = temp_folder_path + "/" + TEMP_FILE_NAME + "2" + AVI_EXTENTION
                    from style_converter_rgb import style_converter
                    with style_converter() as converter:
                        def process2(frame):
                            frame = MMD2illust_util.cv22pil(frame)
                            frame = converter.convert(frame)
                            return frame
                        await self.process_video(temp_video_path1, temp_video_path2, loop, process2, 3, 2)
                    
                    ###### resizing video to (1920, 1080)
                    temp_video_path3 = temp_folder_path + "/" + TEMP_FILE_NAME + "3" + AVI_EXTENTION
                    from waifu2x_processor import waifu2x_processor
                    with waifu2x_processor() as processor:
                        def process3(frame):
                            frame = MMD2illust_util.cv22pil(frame)
                            ###### smaller than (1024, 512)
                            can_pad = MMD2illust_util.get_can_pad(frame.size, original_width, original_height)
                            if (not is_over_sized) and can_pad:
                                real_size = MMD2illust_util.get_pad_real_size((original_width, original_height), frame.width, frame.height)
                                frame = MMD2illust_util.pil_remove_pad(frame, real_size)
                                frame = frame.resize((original_width, original_height))
                                return frame

                            ###### padding video to (1920, 1080)    
                            frame = frame.resize((960, 540))
                            frame = processor.waifu2x_pil(frame)
                            if is_over_sized:
                                return frame
                            else:
                                real_size = MMD2illust_util.get_pad_real_size((original_width, original_height), frame.width, frame.height)
                                frame = MMD2illust_util.pil_remove_pad(frame, real_size)
                                frame = frame.resize((original_width, original_height))
                            return frame
                        await self.process_video(temp_video_path2, temp_video_path3, loop, process3, 3, 3)

                    output_path = name+ILLUST_NAME_EXTENSION+MP4_EXTENTION
                    self.layout.set_progress_text("Extracting audio...")
                    audio_path = temp_folder_path + "/" + TEMP_FILE_NAME + MP3_EXTENTION
                    clip_audio_input = editor.VideoFileClip(input_path).subclip()
                    if clip_audio_input.audio is None:
                        clip_audio_input.close()
                        clip = editor.VideoFileClip(temp_video_path3).subclip()
                        clip.write_videofile(output_path)
                        clip.close()
                    else:
                        clip_audio_input.audio.write_audiofile(audio_path)
                        clip_audio_input.close()
                        self.layout.set_progress_text("Adding audio...")
                        clip = editor.VideoFileClip(temp_video_path3).subclip()
                        clip.write_videofile(output_path, audio=audio_path)
                        clip.close()
            else:
                print("unknown file format")
        loop.stop()
        layout = Main_Layout()
        self.set_layout(layout)

    async def process_video(self, input_video_path, output_video_path, loop, process, num_processes, current_process):
        capture, fps, width, height, total_frame_count, fourcc \
                = MMD2illust_util.get_video_info(input_video_path)
        video_writer = None

        if not capture.isOpened():
            return

        frame_number = 0
        while True:
            is_not_end, frame = capture.read()
            if not is_not_end:
                break
            frame = process(frame)
            
            thread = threading.currentThread()
            if getattr(thread, "cancel_flag", False):
                video_writer.release()
                capture.release()
                cv2.destroyAllWindows()
                loop.stop()
                return

            frame = MMD2illust_util.pil2cv2(frame)
            if video_writer is None:
                width, height = frame.shape[1], frame.shape[0]
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            else: 
                video_writer.write(frame)
            frame_number += 1
            self.layout.set_progress_number(frame_number+1 , total_frame_count, num_processes, current_process)

        video_writer.release()
        capture.release()
        cv2.destroyAllWindows()

    async def process_image(self, input_path, loop):  
        out = Image.open(input_path)
        original_width, original_height = out.size

        ###### padding video to (1024, 576)
        out, is_pad, real_size = MMD2illust_util.pil_pad_to_size(out)
        from waifu2x_processor import waifu2x_processor
        with waifu2x_processor() as processor:
            while (not is_pad):
                out = processor.waifu2x_pil(out)
                out, is_pad, real_size = MMD2illust_util.pil_pad_to_size(out)
                
        ###### converting video
        from style_converter_rgb import style_converter
        with style_converter() as converter:
            out = converter.convert(out)

        ###### smaller than (1024, 512)
        can_pad = MMD2illust_util.get_can_pad(out.size, original_width, original_height)
        if (not is_over_sized) and can_pad:
            out = MMD2illust_util.pil_remove_pad(out, real_size)
            out = out.resize((original_width, original_height))
            return out

        ###### padding video to (1920, 1080)    
        out = out.resize((960, 540))
        with waifu2x_processor() as processor:
            out = processor.waifu2x_pil(out)
        if is_over_sized:
            return out
        else:
            real_size = MMD2illust_util.get_pad_real_size((original_width, original_height), out.width, out.height)
            out = MMD2illust_util.pil_remove_pad(out, real_size)
            out = out.resize((original_width, original_height))
        
        thread = threading.currentThread()
        if getattr(thread, "cancel_flag", False):
            loop.stop()
            return
        return out

    async def expand_illust_and_save(self, input_path, loop):
        global expand_rate
        from waifu2x_processor import waifu2x_processor
        with waifu2x_processor() as waifu2x_processor:
            if os.path.isdir(input_path):
                image_fullnames = [filefullname for filefullname in os.listdir(input_path) if (JPG_EXTENTION in filefullname) | (PNG_EXTENTION in filefullname)]
                total_images_count = len(image_fullnames)
                processed_image_num = 0
                for filefullname in image_fullnames:
                    filefullpath = input_path + "/" + filefullname
                    name, ext = os.path.splitext(filefullpath)
                    output_path = name+EXPAND_NAME_EXTENSION+ext
                    self.layout.set_progress_text("Expanding...")
                    out = waifu2x_processor.waifu2x_path(filefullpath)
                    out = MMD2illust_util.resize_pil_using_rate(out, expand_rate/Expand_Rate)
                    
                    thread = threading.currentThread()
                    if getattr(thread, "cancel_flag", False):
                        loop.stop()
                        return
                        
                    self.layout.set_progress_text("Saving...")
                    out.save(output_path)
                    processed_image_num += 1
                    self.layout.set_progress_number(processed_image_num , total_images_count)
            else:
                name, ext = os.path.splitext(input_path)
                if (ext == JPG_EXTENTION) | (ext == PNG_EXTENTION):
                    output_path = name+EXPAND_NAME_EXTENSION+ext
                    self.layout.set_progress_text("Expanding...")
                    out = waifu2x_processor.waifu2x_path(input_path)
                    out = MMD2illust_util.resize_pil_using_rate(out, expand_rate/Expand_Rate)
                    thread = threading.currentThread()
                    if getattr(thread, "cancel_flag", False):
                        loop.stop()
                        return
                    self.layout.set_progress_text("Saving...")
                    out.save(output_path)
                elif (ext == MP4_EXTENTION) | (ext == AVI_EXTENTION):
                    with tempfile.TemporaryDirectory() as temp_folder_path:
                        print("video")
                        output_path = name+EXPAND_NAME_EXTENSION+MP4_EXTENTION
                        temp_video_path = temp_folder_path + "/" + TEMP_FILE_NAME + AVI_EXTENTION
                        capture = cv2.VideoCapture(input_path)
                        
                        fps = capture.get(cv2.CAP_PROP_FPS)
                        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)*expand_rate)
                        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)*expand_rate)
                        total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

                        if not capture.isOpened():
                            return

                        frame_number = 0
                        while True:
                            is_not_end, frame = capture.read()
                            if not is_not_end:
                                break
                            frame = MMD2illust_util.cv22pil(frame)
                            frame = waifu2x_processor.waifu2x_pil(frame)
                            frame = MMD2illust_util.resize_pil_using_rate(frame, expand_rate/Expand_Rate)
                            thread = threading.currentThread()
                            if getattr(thread, "cancel_flag", False):
                                video_writer.release()
                                capture.release()
                                cv2.destroyAllWindows()
                                loop.stop()
                                return
                            frame = MMD2illust_util.pil2cv2(frame)
                            video_writer.write(frame)
                            frame_number += 1
                            self.layout.set_progress_number(frame_number+1 , total_frame_count)
                        video_writer.release()
                        capture.release()
                        cv2.destroyAllWindows()

                        self.layout.set_progress_text("Extracting audio...")
                        audio_path = temp_folder_path + "/" + TEMP_FILE_NAME + MP3_EXTENTION
                        clip_audio_input = editor.VideoFileClip(input_path).subclip()
                        if clip_audio_input.audio is None:
                            clip_audio_input.close()
                            clip = editor.VideoFileClip(temp_video_path)
                            clip.write_videofile(output_path)
                        else:
                            clip_audio_input.audio.write_audiofile(audio_path)
                            clip_audio_input.close()
                            self.layout.set_progress_text("Adding audio...")
                            clip = editor.VideoFileClip(temp_video_path).subclip()
                            clip.write_videofile(output_path, audio=audio_path)
                            clip.close()
                else:
                    print("unknown file format")
            loop.stop()
            layout = Main_Layout()
            self.set_layout(layout)

    def _on_file_drop(self, window, input_path):
        global is_MMD2illust
        if type(self.layout) is Sub_Layout:
            return
        if type(self.layout) is Process_Layout:
            return
        layout = Process_Layout()
        self.set_layout(layout)
        print(input_path)
        input_path = input_path.decode(UTF8)
        loop = asyncio.get_event_loop()
        loop.stop()
        if is_MMD2illust:
            asyncio.ensure_future(self.mmd2illust_and_save(input_path, loop))
            thread = threading.Thread(target=loop.run_forever)
            thread.start()
            self.process_thread = thread
        else:
            asyncio.ensure_future(self.expand_illust_and_save(input_path, loop))
            thread = threading.Thread(target=loop.run_forever)
            thread.start()
            self.process_thread = thread
            pass
        
    def _on_window_resize(self, *args):
        Window.size = (600, 400)

if __name__ == '__main__':
    if hasattr(sys, '_MEIPASS'):
        resource_add_path(os.path.join(sys._MEIPASS))
    MMD2illust_App().run()