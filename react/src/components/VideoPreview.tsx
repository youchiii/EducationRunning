type VideoPreviewProps = {
  src?: string;
  title: string;
  placeholder?: string;
  className?: string;
};

const VideoPreview = ({
  src,
  title,
  placeholder = "Upload a video to see a preview.",
  className = "",
}: VideoPreviewProps) => {
  return (
    <section className={`space-y-2 ${className}`}>
      <h3 className="text-lg font-semibold">{title}</h3>
      <div className="flex aspect-video w-full items-center justify-center overflow-hidden rounded-lg border border-neutral-300 bg-black/70">
        {src ? (
          <video
            className="h-full w-full object-contain"
            controls
            src={src}
          />
        ) : (
          <p className="px-6 text-center text-sm text-neutral-200">{placeholder}</p>
        )}
      </div>
    </section>
  );
};

export default VideoPreview;
