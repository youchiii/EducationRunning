const TitleBar = () => {
  return (
    <div className="flex items-center justify-between px-3 py-2">
      <div className="flex items-center gap-2">
        <img
          src="/sakuragaoka_logo.jpg"
          alt="Sakuragaoka Analytics"
          className="h-6 w-6 shrink-0 rounded-full"
          draggable={false}
        />
        <span className="text-sm font-medium">Sakuragaoka Analytics</span>
      </div>
      <button className="text-xl leading-none">×</button>
    </div>
  );
};

export default TitleBar;
